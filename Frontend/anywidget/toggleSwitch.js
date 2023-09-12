export function render({ model, el }) {       
    el.classList.add("toggle-switch");
    let label = model.get("label");
    let text = document.createElement("div");
    text.innerHTML = label;
    let toggleSwitch = document.createElement("label");
    toggleSwitch.classList.add("switch");
    toggleSwitch.innerHTML = "<input type='checkbox' id='toggle-switch-checkbox'> <span class='slider round'></span>";
    el.appendChild(text);
    el.appendChild(toggleSwitch);
    let box = document.getElementById("toggle-switch-checkbox");
    box.addEventListener("click", checkHandler);
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