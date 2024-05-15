export function render({ model, el }) {        
    el.classList.add("dataset-display");
    let leftText = document.createElement("div");

    let text = document.createElement("strong");
    text.innerHTML = "Dataset: " + tweetNumber();
    text.classList.add("large-font");

    let fileName = document.createElement("div");
    fileName.innerHTML = model.get("fileName");
    fileName.classList.add("small-font");

    leftText.appendChild(text);
    leftText.appendChild(fileName);
    leftText.classList.add("text-block");

    let changeButton = document.createElement("label");
    changeButton.innerHTML = '<input type="file" id="fileup">Change Dataset'
    changeButton.classList.add("change-input");

    function tweetNumber(){ // might need to display different for large numbers e.g. "1.3 million"
        return model.get("size");
    }

    el.appendChild(leftText);
    el.appendChild(changeButton);
}