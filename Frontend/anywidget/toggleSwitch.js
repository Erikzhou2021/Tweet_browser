export function render({ model, el }) {       
    el.classList.add("toggle-switch");
    let label = model.get("label");
    let text = document.createElement("div");
    text.innerHTML = "<h4> " + label + "<h4/>";
    let toggleSwitch = document.createElement("label");
    toggleSwitch.classList.add("switch");
    let invisibleBox = document.createElement("input");
    invisibleBox.type = "checkbox";
    invisibleBox.addEventListener("click", checkHandler);
    let slider = document.createElement("span");
    slider.classList.add("slider");
    toggleSwitch.appendChild(invisibleBox);
    toggleSwitch.appendChild(slider);

    // toggleSwitch.innerHTML = "<input type='checkbox'> <span class='slider round'></span>";
    el.appendChild(text);
    el.appendChild(toggleSwitch);
    
    if(model.get("value") == 1){
        invisibleBox.checked = true;
    }else{
        invisibleBox.checked = false;
    }

    function checkHandler(){
        if(invisibleBox.checked){
            model.set("value", 1);
            model.save_changes();
        }
        else{
            model.set("value", 0);
            model.save_changes();
        }
    }
}