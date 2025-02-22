export function render({ model, el }) { 
    el.classList.add("sample-selector");
    let buttonText = model.get("label");

    let filePath = model.get("filePath");   
    let select = document.createElement("Select");
    let total = model.get("total");
    let options = model.get("options");
    if(total < options[0]){
        el.classList.add("greyed-out");
    }

    let text = document.createElement("option");
    text.innerHTML = buttonText;
    text.value = -2;
    text.classList.add("hide");
    select.appendChild(text);

    for(var i = 0; i < options.length; i++){
        if(total < options[i]){
            break;
        }
        let option = document.createElement("option");
        option.innerHTML = "\u25EF &nbsp;" + options[i].toString();
        option.value = options[i];
        select.appendChild(option);
    }

    if(total >= options[0]){ // not greyed out
        let option5 = document.createElement("option");
        option5.innerHTML = "\u25EF &nbsp; All";
        option5.value = -1;
        select.appendChild(option5);
    }

    function update(){
        if(select.value == -2){
            return;
        }
        model.set("value", parseInt(select.value));
        model.set("changeSignal", model.get("changeSignal")+1);
        model.save_changes();
        select.value = -2;
    }

    select.onchange = update;

    el.appendChild(select);
}