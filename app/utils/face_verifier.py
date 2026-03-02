import cv2
import random
import numpy as np
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis

MODEL_ROOT = "models"
MODEL_NAME = "auraface"

def ensure_model():
    model_path = Path(MODEL_ROOT) / MODEL_NAME
    if model_path.exists() and any(model_path.glob("*.onnx")):
        return
    print("Downloading AuraFace model...")
    snapshot_download(
        "fal/AuraFace-v1",
        local_dir="models/auraface",
    )
    return

class FaceVerifier:
    def __init__(self, segment_seconds=2, debug=False, conf_threshold=0.75):
        ensure_model()
        self.segment_seconds = segment_seconds
        self.debug = debug
        self.threshold = conf_threshold
        self.faceapp = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            root=".",
        )
        self.faceapp.prepare(ctx_id=0)

    def select_frames_for_extraction(self, video_path: str, num_samples: int = 5, uniform: bool = False) -> List[int]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError("Invalid video metadata")
        
        if uniform:
            indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)
            return sorted(set(indices.tolist()))
        
        segment_frames = int(self.segment_seconds * fps)
        first_segment = list(range(0, min(segment_frames, total_frames)))
        return sorted(random.sample(first_segment, min(num_samples, len(first_segment))))

    def extract_frames(self, video_path, indices):
        cap = cv2.VideoCapture(video_path)
        frames = []
        target_set = set(indices)
        max_index = max(indices)
        current_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_set:
                frames.append(frame)
                if len(frames) == len(indices):
                    break
            if current_idx >= max_index:
                break
            current_idx += 1
        cap.release()
        return frames

    def get_embedding(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.faceapp.get(img)
        if not faces:
            return None
        return faces[0].normed_embedding

    def build_gallery_from_video(self, video_path, num_samples=10):
        indices = self.select_frames_for_extraction(video_path, num_samples)
        frames = self.extract_frames(video_path, indices)
        gallery = []
        for f in frames:
            emb = self.get_embedding(f)
            if emb is not None:
                gallery.append(emb)
        return gallery
    
    def compare_gallery(self, reference_embeddings, candidate_video, num_candidate_sample=10):
        if not reference_embeddings:
            return {
                "result": "NEEDS_REVIEW",
                "match_rate": 0.0,
                "probability": 0.0
            }
        candidate_embeddings = self.build_gallery_from_video(candidate_video, num_samples=num_candidate_sample)
        if not candidate_embeddings:
            return {
                "result": "NEEDS_REVIEW",
                "match_rate": 0.0,
                "probability": 0.0
            }
        scores = []
        true_count = 0
        for emb2 in candidate_embeddings:
            sims = [float(np.dot(np.array(emb1), emb2)) for emb1 in reference_embeddings]
            best_sim = max(sims)
            scores.append(best_sim)
            if best_sim >= self.threshold:
                true_count += 1
        match_rate = true_count / len(candidate_embeddings)
        probability = float(np.mean(scores))
        if match_rate >= 0.5:
            result = "SAME_PERSON"
        elif match_rate == 0:
            result = "DIFFERENT_PERSON"
        else:
            result = "NEEDS_REVIEW"
        return {
            "result": result,
            "match_rate": match_rate,
            "probability": probability,
        }
        
    def identify_faces_and_annotate(self, video_path, all_user_embeddings, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.faceapp.get(rgb)
            for face in faces:
                emb = face.normed_embedding
                bbox = face.bbox.astype(int)
                best_user = "UNKNOWN"
                best_score = -1

                for user_id, gallery in all_user_embeddings.items():
                    sims = [
                        float(np.dot(ref_emb, emb))
                        for ref_emb in gallery
                    ]
                    score = max(sims)
                    if score > best_score:
                        best_score = score
                        best_user = user_id
                x1, y1, x2, y2 = bbox
                label = f"{best_user} ({best_score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(frame)
        cap.release()
        writer.release()

        return output_path