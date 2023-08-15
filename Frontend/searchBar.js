export function render({ model, el }) {        
    el.classList.add("custom-widget");
    let input = document.createElement('input');
    input.value = 'test';
    input.classList.add("search");
    input.addEventListener("keypress", function(event){ if (event.key === "Enter") {addValue(input.value);} });
    let plusButton = document.createElement('span');
    plusButton.innerHTML = "+";
    plusButton.classList.add("plusButton");
    plusButton.addEventListener("click", function(){ addValue(input.value); });
    
    let list = document.createElement('div');

    function addValue(value){
        if(value == ""){
            return;
        }
        for(let index = 0; index < list.children.length; index++){
            if(list.children.item(index).innerHTML == value + '  <span class="close">x</span>'){
                return;
            }
        }
        let currVal = model.get("value");
        model.set("value", currVal.concat(value));
        model.save_changes();

        let tempText = document.createElement('div');
        tempText.innerHTML = value;
        tempText.classList.add("searchedValue");
        tempText.innerHTML = tempText.innerHTML + '&ensp; <span class="close">x</span>';
        list.appendChild(tempText);
        var closebtns = document.getElementsByClassName("close");

        for (let i = 0; i < closebtns.length; i++) {
          closebtns[i].addEventListener("click", function() {
            this.parentElement.remove();
          });
        }
    }
    
    let widget = document.createElement('div');
    widget.appendChild(input);
    widget.appendChild(plusButton);
    widget.appendChild(list);
    el.appendChild(widget);
}