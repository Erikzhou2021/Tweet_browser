export function render({ model, el }) {  
    el.classList.add("dataset-display");      

    let label = document.createElement("label");
    // label.htmlFor = "file-upload";

    let leftText = document.createElement("div");

    let fileName = model.get("fileName");
    if(fileName == null || fileName == ""){
        fileName = "No File Loaded";
    }
    let filePath = model.get("filePath");
    let fileIcon = document.createElement("img");
    fileIcon.src = filePath + "file.svg";

    let uploadIcon = document.createElement("img");
    uploadIcon.src = filePath + "upload.svg";
    
    leftText.innerHTML = "&nbsp; &nbsp; <b><u>" + fileName + "</u></b> &nbsp; " + model.get("size") + " Posts &nbsp; &nbsp;";

    let changeButton = document.createElement("label");
    changeButton.innerHTML = '<input type="file" id="fileup">Change Dataset'
    changeButton.classList.add("change-input");

    
    el.appendChild(fileIcon);
    el.appendChild(leftText);
    el.appendChild(uploadIcon);
    // el.appendChild(label);
}