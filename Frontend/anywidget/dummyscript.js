export function render({ model, el }) {   
    const fontLink = 'https://fonts.googleapis.com/css?family=Roboto';
    const existingLink = document.querySelector(`link[href="${fontLink}"]`);
    if (!existingLink) {
      const linkTag = document.createElement('link');
      linkTag.rel = 'stylesheet';
      linkTag.href = fontLink;
      document.head.appendChild(linkTag);
    } 
    

    let button = document.querySelector('.search-button');
    function preSearch(){
        let searchBars = document.querySelectorAll('.plusButton');
        searchBars.forEach((elem) =>{
            elem.click();
        });
        let invisButton = document.querySelector('.hidden-button');
        invisButton.click();
    } 
    if(button != null){
        button.addEventListener("click", preSearch);
    }

    const query = '.date-constraint > input:first-of-type';
    let start = model.get("calendarStart");
    let end = model.get("calendarEnd");
    let results = document.querySelectorAll(query);
    results.forEach((calenderEl) => {
        calenderEl.setAttribute('min', start);
        calenderEl.setAttribute('max', end);
    });




    let fileUp = document.querySelector(".widget-upload");
    if(fileUp == null){
        return;
    }
    if(document.getElementById("file-upload-cover") != null){
        return;
    }
    fileUp.setAttribute("id", "file-upload");
    let parent = fileUp.parentElement;

    let label = document.createElement("label");
    label.classList.add("dataset-display");     
    label.setAttribute("id", "file-upload-cover"); 

    label.htmlFor = "file-upload";

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



    function confirmChanges(){
        let response = confirm("Apply filters?\nYou have made changes to the filter settings, but they have not been applied yet.");
        let responseCode = 0;
        if(response){
            responseCode = 1;
        }
        model.set("userResponse", responseCode);
        model.set("changeSignal", model.get("changeSignal") + 1);
        model.save_changes();
    }
    model.on("change:alertTrigger", confirmChanges);

    
    label.appendChild(fileIcon);
    label.appendChild(leftText);
    label.appendChild(uploadIcon);
    parent.appendChild(label);
}