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
   
    function checkHandler(){
        if(box.checked){
            model.set("label", 1);
            model.save_changes();
        }
        else{
            model.set("label", 0);
            model.save_changes();
        }
    }
}