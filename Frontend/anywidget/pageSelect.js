export function render({ model, el }) {       
    let filePath = model.get("filePath");
    el.classList.add("page-select");     

    let left = document.createElement("img");
    left.src = filePath + "left.svg";
    left.classList.add("arrow");
    left.addEventListener("click", decrement);

    let container = document.createElement("div");
    container.classList.add("page-input");


    let text1 = document.createElement("p");
    text1.innerHTML = "Page &nbsp;";
    
    let input = document.createElement("input");
    input.type = "number";
    input.value = "1";
    input.classList.add("page-num");
    input.addEventListener("input", function(){
        pageChange(input.value);
    });
    input.addEventListener("blur", resetPage);

    let text2 = document.createElement("p");
    text2.innerHTML = "&nbsp; out of " + model.get("maxPage");

    container.appendChild(text1);
    container.appendChild(input);
    container.appendChild(text2);

    let right = document.createElement("img");
    right.src = filePath + "right.svg";
    right.classList.add("arrow");
    right.addEventListener("click", increment);

    model.on("change:maxPage", update);

    function pageChange(newVal){
        if(newVal == null || newVal == ""){
            return;
        }
        let maxPage = model.get("maxPage");
        if(newVal < 1){
            newVal = 1;
        }
        else if(newVal > maxPage){
            newVal = maxPage;
        }
        input.value = newVal;
        model.set("value", newVal);
        model.save_changes();
    }

    function resetPage(hardReset = false){
        if(input.value == null || input.value == "" || hardReset){
            input.value = model.get("value");
        }
    }

    function increment(){
        let maxPage = model.get("maxPage");
        if(input.value <= maxPage-1){
            input.value++;
            model.set("value", input.value);
            let signal = model.get("changeSignal");
            model.set("changeSignal", signal + 1);
            model.save_changes();
        }
    }

    function decrement(){
        if(input.value > 1){
            input.value--;
            model.set("value", input.value);
            let signal = model.get("changeSignal");
            model.set("changeSignal", signal + 1);
            model.save_changes();
        }
    }

    function update(){
        let maxPage = model.get("maxPage");
        if(model.get("value") < 1){
            model.set("value", 1);
            model.save_changes();
            input.value = 1;
        }
        else if(model.get("value") > maxPage){
            model.set("value", maxPage);
            model.save_changes();
            input.value = String(maxPage);
        }
        else{
            input.value = String(model.get("value"))
        }
        text2.innerHTML = "&nbsp; out of " + maxPage;
    }

    resetPage(true);
    el.appendChild(left);
    el.appendChild(container);
    el.appendChild(right);
}