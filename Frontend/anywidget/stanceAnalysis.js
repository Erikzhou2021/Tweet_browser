export function render({ model, el }) {  
    let infoPage = createAndAdd(el, "", "info-page"); 
    createAndAdd(infoPage, "Semantic Search and Stance Annotation", "info-title");
    createAndAdd(infoPage, "Stance Analysis", "");
    createAndAdd(infoPage, "Stance Analysis performs machine-based analysis on the most relevant 200 posts to the <b>Semantic Search</b> (Highest similarity score) from a semantic search topic, categorizing them into up to four distinct stances based on the following steps. Providing examples for each stance helps the model refine the analysis and deliver more accurate stance labels for the results.", "info-text");
    let nextButton = document.createElement("button");
    nextButton.innerHTML = "Next";
    nextButton.classList.add("generic-button");
    infoPage.appendChild(nextButton);
    el.classList.add("stance-analysis");

    nextButton.addEventListener("click", chagePage);
    if(model.get("pageNumber") == 0){
        changePage();
    }

    function createAndAdd(parent, html, cssClass){
        let temp = document.createElement("div");
        temp.innerHTML = html;
        if(cssClass != ""){
            temp.classList.add(cssClass);
        }
        parent.appendChild(temp);
        return temp;
    }

    function createInputs(inputNumber, parentContainer){
        let tempContainer = createAndAdd(parentContainer, "", "input-container2");
        createAndAdd(tempContainer, "Stance &nbsp;" + inputNumber, "body0");
        let stanceInput = document.createElement("input");
        tempContainer.appendChild(stanceInput);

        let tempContainer2 = createAndAdd(parentContainer, "", "input-container2");
        createAndAdd(tempContainer2, "Stance &nbsp;" + inputNumber + "&nbsp examples", "body0");
        let exampleInput = document.createElement("input");
        tempContainer2.appendChild(exampleInput);
    }

    function chagePage(){
        model.set("pageNumber", 0);
        model.save_changes();
        el.innerHTML = '';
        createAndAdd(el, "Step1 - Define Stance", "info-title");
        let userInputContainer = createAndAdd(el, "", "input-page-container");
        let inputContainer1 = createAndAdd(userInputContainer, "", "input-container1");
        createAndAdd(inputContainer1, "Stance Topic", "body0");
        let stanceTopicInput = document.createElement("input");
        inputContainer1.appendChild(stanceTopicInput);

        for(var i = 0; i < 4; i++){
            createInputs(i+1, userInputContainer);
        }

        let applyButton = document.createElement("button");
        applyButton.innerText = "Apply Stance Annotation";
        applyButton.classList.add("generic-button");
        applyButton.style.marginLeft = "auto";
        el.appendChild(applyButton);
    }
}