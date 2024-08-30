export function render({ model, el }) {        
    el.classList.add("dataset-display");
    let leftText = document.createElement("div");

    let fileName = model.get("fileName");
    if(fileName == null || fileName == ""){
        fileName = "No File Loaded";
    }
    let filePath = model.get("filePath");
    let fileIcon = document.createElement("img");
    fileIcon.src = filePath + "file.svg";
    
    leftText.innerHTML = "&nbsp; <b><u>" + fileName + "</u></b> &nbsp; " + model.get("size") + " Posts";

    let changeButton = document.createElement("label");
    changeButton.innerHTML = '<input type="file" id="fileup">Change Dataset'
    changeButton.classList.add("change-input");

    el.appendChild(fileIcon);
    el.appendChild(leftText);
    // el.appendChild(changeButton);
}