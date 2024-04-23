export function render({ model, el }) { 
    el.classList.add("sample-selector");

    let filePath = model.get("filePath");   
    let select = document.createElement("Select");
    // button.innerHTML = "Generate New Sample >";
    // button.classList.add("generic-button");

    let text = document.createElement("option");
    text.innerHTML = "Generate New Sample >";
    text.value = -2;
    text.classList.add("hide");
    select.appendChild(text);

    let option1 = document.createElement("option");
    option1.innerHTML = "50";
    option1.value = 50;
    select.appendChild(option1);

    let option2 = document.createElement("option");
    option2.innerHTML = "100";
    option2.value = 100;
    select.appendChild(option2);

    let option3 = document.createElement("option");
    option3.innerHTML = "150";
    option3.value = 150;
    select.appendChild(option3);

    let option4 = document.createElement("option");
    option4.innerHTML = "200";
    option4.value = 200;
    select.appendChild(option4);

    let option5 = document.createElement("option");
    option5.innerHTML = "All";
    option5.value = -1;
    select.appendChild(option5);

    function update(){
        if(select.value == -2){
            return;
        }
        model.set("value", parseInt(select.value));
        // model.save_changes(); // required so model observe doesn't get confused
        model.set("changeSignal", model.get("changeSignal")+1);
        model.save_changes();
        select.value = -2;
    }

    select.onchange = update;

    el.appendChild(select);
}