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
    
    leftText.innerHTML = "&nbsp; &nbsp; <b><u>" + fileName + "</u></b> &nbsp; " + model.get("size").toLocaleString() + " Posts &nbsp; &nbsp;";



    function confirmChanges(){
        let response = confirm("Apply filters?\nYou have made changes to the filter settings, but they have not been applied yet.");
        let responseCode = 0;
        if(response){
            if(isLoading){
                let comfirmation = confirm("Apply Refine Results modification?\nStance Annotation is running in the background. Modifying the Refine Results selections will end Stance Annotation. If you want to continue Refining Results, click OK.");
                if(comfirmation){
                    responseCode = 1;
                }
            }
            else{
                responseCode = 1;
            }
        }
        model.set("userResponse", responseCode);
        model.set("changeSignal", model.get("changeSignal") + 1);
        model.save_changes();
    }
    model.on("change:alertTrigger", confirmChanges);

    function doComfirmation(prompt, varName){
        let responseCode = 1;
        if(isLoading){
            let response = confirm(prompt);
            if(!response){
                responseCode = 0;
            }
        }
        model.set("userResponse", responseCode);
        model.set(varName, model.get(varName) + 1);
        model.save_changes();
    }
    function comfirmSearch(){
        let prompt = "Attempt to modify Search Criteria?\nStance Annotation is running in the background. Modifying Search Criteria will end Stance Annotation. If you want to continue modifying Search Criteria, click OK.";
        doComfirmation(prompt, "searchChangeSignal");
    }
    function confirmFilter(){
        let prompt = "Apply Refine Results modification?\nStance Annotation is running in the background. Modifying the Refine Results selections will end Stance Annotation. If you want to continue Refining Results, click OK.";
        doComfirmation(prompt, "filterChangeSignal");
    }
    model.on("change:searchTrigger", comfirmSearch);
    model.on("change:filterTrigger", confirmFilter);


    label.appendChild(fileIcon);
    label.appendChild(leftText);
    label.appendChild(uploadIcon);
    parent.appendChild(label);

    let isLoading = false;
    function updateLoading(){
        if(model.get("activeStanceAnalysis") == 1){
            isLoading = true;
        }
        else{
            isLoading = false;
        }
    }
    model.on("change:activeStanceAnalysis", updateLoading);
    // function shutdownKernel(){ // from .local/share/jupyter/voila/templates/base/static/main.js
    //     const matches = document.cookie.match('\\b_xsrf=([^;]*)\\b');
    //     const xsrfToken = (matches && matches[1]) || '';
    //     const configData = JSON.parse(document.getElementById('jupyter-config-data').textContent);
    //     const baseUrl = configData.baseUrl;
    //     const data = new FormData();
    //     data.append("_xsrf", xsrfToken);
    //     window.navigator.sendBeacon(`${baseUrl}voila/api/shutdown/${kernel.id}`, data);
    //     // kernel.dispose();
    // }
    window.addEventListener("beforeunload", (event) => {
        if (isLoading) {
            event.preventDefault();
            event.returnValue = "";
        }
    });
}