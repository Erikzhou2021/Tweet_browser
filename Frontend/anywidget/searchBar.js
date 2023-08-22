export function render({ model, el }) {   
    let total = 0

    let inputBar = document.createElement("div");
    inputBar.classList.add("search-bar");
    let input = document.createElement('input');
    input.value = '';
    input.classList.add("search");
    input.addEventListener("keypress", function(event){ if (event.key === "Enter") {addValue(input.value);} });
    let plusButton = document.createElement('span');
    plusButton.innerHTML = "+";
    plusButton.classList.add("plusButton");
    plusButton.addEventListener("click", function(){ addValue(input.value); });
    
    let list = document.createElement('div');
    list.classList.add("value-list");
    let closeTag = '<span class="close">x</span>';

    function addValue(value){
        if(value == ""){
            return;
        }
        for(let index = 0; index < list.children.length; index++){
            if(list.children.item(index).innerHTML == value + closeTag){
                return;
            }
        }
        let currVal = model.get("value");
        total++;
        model.set("value", currVal.concat(value));
        model.set("totalItems", total);
        model.save_changes();


        let tempText = document.createElement('div');
        tempText.innerHTML = value;
        tempText.classList.add("searched-value");
        tempText.innerHTML = tempText.innerHTML + closeTag;
        list.appendChild(tempText);
        var closebtns = document.getElementsByClassName("close");

        for (let i = 0; i < closebtns.length; i++) {
          closebtns[i].addEventListener("click", removeValue);
        }
    }

    function removeValue(){
        let deletedVal = this.parentElement.innerHTML;
            deletedVal = deletedVal.substring(0, deletedVal.length - closeTag.length);
            let oldVal = model.get("value");
            for(let i = 0; i < oldVal.length; i++){
                if(oldVal[i] == deletedVal){
                    oldVal[i] = oldVal[oldVal.length -1];
                    oldVal.pop();
                    total--;
                    //model.set("value", [...oldVal]);
                    // need to do this weird stuff, might be a bug in backbone.js or anywidget
                    // still doesn't even work properly when deleting the last element
                    model.set("value", []);
                    model.set("value", oldVal);
                    model.set("totalItems", total);
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