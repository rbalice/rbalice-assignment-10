document.getElementById("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "<p>Loading...</p>";

    const formData = new FormData();
    const text = document.getElementById("text").value;
    const image = document.getElementById("image").files[0];
    const weight = document.getElementById("weight").value;
    const embeddingType = document.getElementById("embedding-type").value;

    if (text) formData.append("text", text);
    if (image) formData.append("image", image);
    formData.append("weight", weight);
    formData.append("embedding_type", embeddingType);

    const response = await fetch("/search", {
        method: "POST",
        body: formData
    });

    const results = await response.json();
    // const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    results.forEach(([fileName, similarity]) => {
        const resultContainer = document.createElement("div");
        resultContainer.style.display = "inline-block";
        resultContainer.style.textAlign = "center";
        resultContainer.style.margin = "10px";

        const img = document.createElement("img");
        img.src = `/coco_images_resized/${fileName}`; 
        img.alt = `Similarity: ${similarity}`;
        img.title = `Similarity: ${similarity.toFixed(4)}`;
        img.style.width = "150px";
        img.style.margin = "10px";
        resultsDiv.appendChild(img);

        const scoreSpan = document.createElement("span");
        scoreSpan.textContent = `Similarity: ${similarity.toFixed(4)}`;
        scoreSpan.style.display = "block";
        scoreSpan.style.marginTop = "5px";

        resultContainer.appendChild(img);
        resultContainer.appendChild(scoreSpan);
        resultsDiv.appendChild(resultContainer);
    });    
});
