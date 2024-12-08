from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open_clip 
from open_clip import create_model_and_transforms, tokenizer
import pandas as pd
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

image_folder = "coco_images_resized" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_df = pd.read_pickle('image_embeddings.pickle')  # Load precomputed embeddings
pca_df = pd.read_pickle('pca_embeddings.pickle')
# Load PCA model and CLIP model
with open('pca_model.pickle', 'rb') as f:
    pca_model = pickle.load(f)
with open('pca_scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)
# with open('pca_embeddings.pickle', 'rb') as f:
#     pca_embeddings = pickle.load(f)
# pca_df = pd.DataFrame({
#     'filename': pca_embeddings['filename'],
#     'embedding': np.array(pca_embeddings['embedding'])
# })


# Load model (改)
model, _, preprocess = create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
model = model.to(device)
model.eval()

try:
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
except Exception as e:
    print(f"Error getting tokenizer: {e}")
    # Fallback tokenization method
    def tokenizer(texts):
        return torch.cat([open_clip.tokenize(text) for text in texts])

def find_similar_images(query_embedding, embedding_type='clip', top_n=5):
    """Find the top N most similar images."""
    if embedding_type == 'clip':
        similarities = []
        for index, row in clip_df.iterrows():
            # Ensure embeddings are numpy arrays
            stored_embedding = row["embedding"]
            similarity = float(np.dot(query_embedding, stored_embedding))
            similarities.append((row["file_name"], similarity))
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    else:  # PCA
        embeddings = np.vstack(pca_df['embedding'].values) 

        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        nearest_indices = np.argsort(distances)[:top_n]

        results = [(pca_df.iloc[i]['filename'], float(distances[i])) for i in nearest_indices]
        return results

@app.route('/')
def index():
    return render_template('index.html')

# Serve files from coco_images_resized
@app.route('/coco_images_resized/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'coco_images_resized'), filename)

@app.route('/search', methods=['POST'])
def search():
    # Handle text input
    text_embedding = None
    image_embedding = None

    text_query = request.form.get("text","").strip()
    weight = float(request.form.get("weight", 0.8))  # Default weight
    embedding_type = request.form.get("embedding_type", "clip")

    if text_query:
        if embedding_type == 'clip':
            text_token = tokenizer([text_query])
            with torch.no_grad():
                text_embedding = F.normalize(model.encode_text(text_token)).detach().cpu().numpy()
    
    # Handle image input
    if "image" in request.files:
        image_file = request.files["image"]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)
        if embedding_type == 'pca':
            image = Image.open(image_path).convert("RGB")
            resized_image = image.resize((224, 224))
            img = np.asarray(resized_image, dtype=np.float32) / 255.0
            img = img.flatten()

            # image_tensor = np.array(Image.open(image_path).convert("RGB")).reshape(1, -1)
            scaled_image = scaler.transform(img.reshape(1, -1))
            # scaled_image = scaler.transform(image_tensor)

            pca_image_embedding = pca_model.transform(scaled_image)
            image_embedding = pca_image_embedding
        else:
            # Open and preprocess the image
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_image_embedding = F.normalize(model.encode_image(image)).detach().cpu().numpy()
            image_embedding = clip_image_embedding

    # Combine queries if both exist
    # lam = 0.8  # Adjust weight
    query_embedding = None
    if embedding_type == 'pca':
        query_embedding = image_embedding
    else:
        if text_embedding is not None and image_embedding is not None:
            # Weighted combination of text and image embeddings
            query_embedding = F.normalize(torch.from_numpy(weight * text_embedding + (1 - weight) * image_embedding)).cpu().numpy()
        elif text_embedding is not None:
            query_embedding = text_embedding
        elif image_embedding is not None:
            query_embedding = image_embedding
    
    if query_embedding is None:
            return jsonify({"error": "No valid query (text or image) provided"}), 400

    # Retrieve most similar images
    results = find_similar_images(query_embedding[0], embedding_type)  # query_embedding[0] to pass single vector
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
