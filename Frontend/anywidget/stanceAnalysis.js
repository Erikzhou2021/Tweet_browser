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

    const NUM_INPUTS = 4;

    nextButton.addEventListener("click", changePage);

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
        stanceInput.autocomplete = "off";
        stanceInput.value = model.get("stances")[inputNumber];
        stanceInput.id = "stanceInput" + inputNumber.toString();
        tempContainer.appendChild(stanceInput);

        let tempContainer2 = createAndAdd(parentContainer, "", "input-container2");
        createAndAdd(tempContainer2, "Stance &nbsp;" + inputNumber + "&nbsp examples", "body0");
        let exampleInput = document.createElement("input");
        exampleInput.autocomplete = "off";
        exampleInput.value = model.get("examples")[inputNumber];
        exampleInput.id = "exampleInput" + inputNumber.toString();
        tempContainer2.appendChild(exampleInput);
    }

    function changePage(){
        model.set("pageNumber", 0);
        model.save_changes();
        el.innerHTML = '';
        createAndAdd(el, "Step1 - Define Stance", "info-title");
        let userInputContainer = createAndAdd(el, "", "input-page-container");
        let inputContainer1 = createAndAdd(userInputContainer, "", "input-container1");
        createAndAdd(inputContainer1, "Stance Topic", "body0");
        let stanceTopicInput = document.createElement("input");
        stanceTopicInput.autocomplete = "off";
        stanceTopicInput.id = "stanceTopicInput";
        stanceTopicInput.value = model.get("topic");
        inputContainer1.appendChild(stanceTopicInput);

        for(var i = 0; i < NUM_INPUTS; i++){
            createInputs(i, userInputContainer);
        }

        let applyButton = document.createElement("button");
        applyButton.innerText = "Apply Stance Annotation";
        applyButton.classList.add("generic-button");
        applyButton.style.marginLeft = "auto";
        applyButton.addEventListener("click", changeToResultsPage);
        el.appendChild(applyButton);
    }

    if(model.get("pageNumber") > -1){
        changePage();
    }

    function changeToResultsPage(){
        model.set("topic", document.getElementById("stanceTopicInput").value);
        let stances = [];
        let examples = [];
        for(var i = 0; i < NUM_INPUTS; i++){
            stances.push(document.getElementById("stanceInput" + i.toString()).value);
            examples.push(document.getElementById("exampleInput" + i.toString()).value);
        }
        model.set("stances", stances);
        model.set("examples", examples);
        model.set("pageNumber", 1);
        model.save_changes();
    }
}