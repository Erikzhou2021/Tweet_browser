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
    }
    else{
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

    const query = '.date-constraint > input:first-of-type';
    let start = model.get("calendarStart");
    let end = model.get("calendarEnd");
    let results = document.querySelectorAll(query);
    results.forEach((calenderEl) => {
        calenderEl.setAttribute('min', start);
        calenderEl.setAttribute('max', end);
    });

    let button = document.querySelector('.search-button');
    button.addEventListener("click", preSearch);
    function preSearch(){
        let searchBars = document.querySelectorAll('.plusButton');
        searchBars.forEach((elem) =>{
            elem.click();
        });
        let invisButton = document.querySelector('.hidden-button');
        invisButton.click();
    }
    
}