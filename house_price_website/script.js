const API_URL = "http://127.0.0.1:8000";


document.getElementById("predictForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let formData = {
        latitude: parseFloat(document.getElementById("latitude").value),
        longitude: parseFloat(document.getElementById("longitude").value),
        area_sqft: parseFloat(document.getElementById("area").value),
        bedrooms: parseInt(document.getElementById("bedrooms").value),
        bathrooms: parseInt(document.getElementById("bathrooms").value)
    };

    document.getElementById("result").innerHTML = "⏳ Predicting...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        
        document.getElementById("result").innerHTML =
            `💰 Estimated Price: <b>₹ ${data.predicted_price.toLocaleString()}</b>`;
    } 
    catch (error) {
        document.getElementById("result").innerHTML = "❌ Error contacting the server!";
        console.error(error);
    }
});
