export function render({ model, el }) { 
    el.classList.add("sample-selector");

    let filePath = model.get("filePath");   
    let select = document.createElement("Select");
    let total = model.get("total");
    if(total < 50){
        el.classList.add("greyed-out");
    }

    let text = document.createElement("option");
    text.innerHTML = "Generate New Sample >";
    text.value = -2;
    text.classList.add("hide");
    select.appendChild(text);

    if(total >= 50){
        let option1 = document.createElement("option");
        option1.innerHTML = "\u25EF &nbsp; 50";
        option1.value = 50;
        select.appendChild(option1);
    }

    if(total >= 100){
        let option2 = document.createElement("option");
        option2.innerHTML = "\u25EF &nbsp; 100";
        option2.value = 100;
        select.appendChild(option2);
    }

    if(total >= 150){
        let option3 = document.createElement("option");
        option3.innerHTML = "\u25EF &nbsp; 150";
        option3.value = 150;
        select.appendChild(option3);
    }

    if(total >= 200){
        let option4 = document.createElement("option");
        option4.innerHTML = "\u25EF &nbsp; 200";
        option4.value = 200;
        select.appendChild(option4);
    }

    if(total >= 50){ // not greyed out
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