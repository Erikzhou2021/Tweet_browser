export function render({ model, el }) {        
    // el.classList.add("dataset-display");
    let leftText = document.createElement("div");

    let text = document.createElement("strong");
    text.innerHTML = "Dataset: " + tweetNumber();
    text.classList.add("large-font");

    let fileName = document.createElement("div");
    let file = model.get("fileName");
    if(file == null || file == ""){
        file = "No File Loaded";
    }
    fileName.innerHTML = file;
    fileName.classList.add("small-font");

    el.appendChild(text);
    el.appendChild(fileName);

    function tweetNumber(){ // might need to display different for large numbers e.g. "1.3 million"
        return model.get("size");
    }
}