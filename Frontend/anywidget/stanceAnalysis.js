export function render({ model, el }) {  
    let infoPage = createAndAdd(el, "", "info-page"); 
    createAndAdd(infoPage, "Semantic Search and Stance Annotation", "heading4").classList.add("medium");
    createAndAdd(infoPage, "Stance Analysis", "");
    createAndAdd(infoPage, "Stance Analysis performs machine-based analysis on the most relevant 200 posts to the <b>Semantic Search</b> (Highest similarity score) from a semantic search topic, categorizing them into up to four distinct stances based on the following steps. Providing examples for each stance helps the model refine the analysis and deliver more accurate stance labels for the results.", "info-text");
    let nextButton = document.createElement("button");
    nextButton.innerHTML = "Next";
    nextButton.classList.add("generic-button");
    nextButton.addEventListener("click", dismissInfo);
    infoPage.appendChild(nextButton);
    el.classList.add("stance-analysis");    

    let applyButton = document.createElement("button"); // might break something
    applyButton.classList.add("greyed-out");
    const NUM_INPUTS = 4;

    function createAndAdd(parent, html, cssClass){
        let temp = document.createElement("div");
        temp.innerHTML = html;
        if(cssClass != ""){
            temp.classList.add(cssClass);
        }
        parent.appendChild(temp);
        return temp;
    }

    function showWarningPage(){
        el.innerHTML = "";
        let warningPage = createAndAdd(el, "", "info-page"); 
        createAndAdd(warningPage, "Suggestion", "heading4").classList.add("medium");
        createAndAdd(warningPage, "Using Semantic Match or Exact Match (in Searched Criteria) will help narrow the results and likely improve the stance annotation. Do you want to proceed with the Stance Analysis with the current set of results?", "info-text");
        let buttons = createAndAdd(warningPage, "", "info-page-buttons");
        let modifyButton = document.createElement("button");
        modifyButton.classList.add("modify-search-button")
        modifyButton.innerHTML = "Modify Searched Criteria";
        modifyButton.addEventListener("click", ()=>{
            model.set("changeSignal", model.get("changeSignal") + 1);
            model.save_changes();
        });
        buttons.appendChild(modifyButton);
        let proceedButton = document.createElement("button");
        proceedButton.classList.add("generic-button");
        proceedButton.innerHTML = "Proceed to Stance Analysis";
        proceedButton.addEventListener("click", ()=>{
            model.set("pageNumber", 0);
            model.save_changes();
        });
        buttons.appendChild(proceedButton);
    }

    function checkFilled(){
        let numFilled = 0;
        for(var i = 0; i < NUM_INPUTS; i++){
            let temp = document.getElementById("stanceInput" + i.toString());
            if(temp == null){
                if(model.get("stances")[i] != ""){
                    numFilled++;
                }
                break;
            }
            if(temp.value != ""){
                numFilled++;
            }
        }
        let temp2 = document.getElementById("stanceTopicInput");
        if(temp2 == null){
            if(model.get("topic") == ""){
                applyButton.classList.add("greyed-out");
            }
            else{
                applyButton.classList.remove("greyed-out");
            }
            return;
        }
        if((temp2.value != "") && numFilled >= 2){
            applyButton.classList.remove("greyed-out");
        }
        else{
            applyButton.classList.add("greyed-out");
        }
    }

    function setColor(event){
        let element = event.target;
        let id = element.id;
        const prefix = "colorPicker";
        let num = id.substring(prefix.length);
        let titleContainer = document.getElementById("titleContainer" + num);
        titleContainer.style.backgroundColor = element.value;
        let oldVals = model.get("colors");
        const root = document.documentElement;
        root.style.setProperty('--stanceColor' + num, element.value);
        oldVals[parseInt(num)] = element.value;
        model.set("colors", oldVals);
        model.save_changes();
    }

    function createInputs(inputNumber, parentContainer){
        let tempContainer = createAndAdd(parentContainer, "", "input-container2");
        let titleContainer = createAndAdd(tempContainer, "", "color-picker");
        titleContainer.id = "titleContainer" + inputNumber.toString();
        let colorPicker = document.createElement("input");
        colorPicker.id = "colorPicker" + inputNumber.toString();
        colorPicker.type = "color";
        colorPicker.addEventListener("blur", setColor);
        colorPicker.value = model.get("colors")[inputNumber];
        titleContainer.style.backgroundColor = colorPicker.value
        titleContainer.appendChild(colorPicker);
        let temp = createAndAdd(titleContainer, "Stance &nbsp;" + String(inputNumber + 1), "body0");
        temp.classList.add("medium");
        let icon = document.createElement("img");
        icon.src = model.get("filePath") + "colorize.svg";
        titleContainer.appendChild(icon);
        createAndAdd(tempContainer, "Opinion", "body3");
        let stanceInput = document.createElement("input");
        stanceInput.autocomplete = "off";
        stanceInput.value = model.get("stances")[inputNumber];
        stanceInput.addEventListener("input", checkFilled);
        stanceInput.id = "stanceInput" + inputNumber.toString();
        tempContainer.appendChild(stanceInput);

        createAndAdd(tempContainer, "Examples", "body3").classList.add("stance-extra-margin");
        let exampleInput = document.createElement("input");
        exampleInput.classList.add("stance-example");
        exampleInput.autocomplete = "off";
        exampleInput.value = model.get("examples")[inputNumber];
        exampleInput.id = "exampleInput" + inputNumber.toString();
        tempContainer.appendChild(exampleInput);
    }

    function changePage(){
        if(model.get("seenInfo") < 1){
            return;
        }
        if(model.get("pageNumber") == -1){
            showWarningPage();
            return;
        }
        el.innerHTML = '';
        createAndAdd(el, "Define Topic & Stances *", "heading4").classList.add("medium");
        createAndAdd(el, '*Posts not fitting in provided stances will be labeled as “Irrelevant”', "bodye2").style.marginTop = "-8px";
        let userInputContainer = createAndAdd(el, "", "input-page-container");
        let inputContainer1 = createAndAdd(userInputContainer, "", "input-container1");
        createAndAdd(inputContainer1, "Stance Topic", "body0");
        let stanceTopicInput = document.createElement("input");
        stanceTopicInput.autocomplete = "off";
        stanceTopicInput.id = "stanceTopicInput";
        stanceTopicInput.value = model.get("topic");
        stanceTopicInput.addEventListener("input", checkFilled);
        inputContainer1.appendChild(stanceTopicInput);

        for(var i = 0; i < NUM_INPUTS; i++){
            createInputs(i, userInputContainer);
        }

        applyButton.innerText = "Apply Stance Annotation";
        applyButton.classList.add("generic-button");
        applyButton.style.marginLeft = "auto";
        applyButton.addEventListener("click", changeToResultsPage);
        el.appendChild(applyButton);
        checkFilled();
    }

    function dismissInfo(){
        model.set("seenInfo", 1);
        model.save_changes();
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
    model.on("change:pageNumber", changePage);
    changePage();
}