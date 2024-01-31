export function render({ model, el }) {   
    let headerText = model.get("header");
    if(headerText != null && headerText != ""){
        let header = document.createElement("h4");
        header.innerHTML = headerText;
        el.appendChild(header);
    }
    let inputBar = document.createElement("div");
    inputBar.classList.add("search-bar");
    let input = document.createElement('input');
    let placeholder = model.get("placeholder");
    if(placeholder != null && placeholder != ""){
        input.placeholder = placeholder;
    }
    input.value = '';
    input.classList.add("search");
    input.addEventListener("keypress", function(event){ if (event.key === "Enter") {addValue(input.value);} });
    let plusButton = document.createElement('span');
    plusButton.innerHTML = "+";
    plusButton.classList.add("plusButton");
    plusButton.addEventListener("click", function(){ addValue(input.value); });
    
    let list = document.createElement('div');
    list.classList.add("value-list");

    let currVal = model.get("value");
    let count = model.get("count");
    for(let i = 0; i < count; i++){
        createSearchedValue(currVal[i]);
    }

    function addValue(value){
        if(value == ""){
            return;
        }
        let keywords = value.match(/(?:[^\s"]+|"[^"]*")+/g);
        for(let word of keywords){
            word = word.replace(/['"]+/g, '')
            let currVal = model.get("value");
            for(let index = 0; index < currVal.length; index++){
                if(currVal[index] == word){
                    return;
                }
            }
            let currCount = model.get("count");
            model.set("count", currCount+1);
            model.set("value", currVal.concat(word));
            model.save_changes();

            input.value = "";
            createSearchedValue(word);   
        }     
    }

    function createSearchedValue(value){
        let tempText = document.createElement('div');
        let closeTag = document.createElement('span');
        closeTag.innerHTML = "x";
        closeTag.addEventListener("click", removeValue);
        tempText.innerHTML = value;
        tempText.classList.add("searched-value");
        tempText.appendChild(closeTag);
        list.appendChild(tempText);
    }

    function removeValue(){
        let closeHTML = "<span>x</span>";
        let deletedVal = this.parentElement.innerHTML;
        deletedVal = deletedVal.substring(0, deletedVal.length - closeHTML.length);
        let oldVal = model.get("value");
        for(let i = 0; i < oldVal.length; i++){
            if(oldVal[i] == deletedVal){
                oldVal[i] = oldVal[oldVal.length -1];
                oldVal.pop();
                //model.set("value", [...oldVal]);
                // need to do this weird stuff, might be a bug in backbone.js or anywidget
                // still doesn't even work properly when deleting the last element
                let currCount = model.get("count");
                model.set("count", currCount-1);
                model.set("value", []);
                model.set("value", oldVal);
                model.save_changes();
                break;
            }
        }
        this.parentElement.remove();
        model.save_changes();
    }

    inputBar.appendChild(input);
    inputBar.appendChild(plusButton);
    el.appendChild(inputBar);
    el.appendChild(list);
}