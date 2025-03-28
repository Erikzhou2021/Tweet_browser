export function render({ model, el }) { 
    el.classList.add("typical-sample-selector");

    function reload(){
        el.innerHTML = "";
        if(model.get("visible") == 0){
            return;
        }
        let select = document.createElement("Select");
        let total = model.get("total");
        let options = model.get("options");
        let text = document.createElement("option");
        text.innerHTML = Math.min(total, model.get("value"));
        text.value = -2;
        text.classList.add("hide");
        select.appendChild(text);
        select.value = text.value;

        for(var i = 0; i < options.length; i++){
            if(total < options[i]){
                break;
            }
            let option = document.createElement("option");
            option.innerHTML = "\u25EF &nbsp;" + options[i].toString();
            option.value = options[i];
            select.appendChild(option);
        }

        function update(){
            if(select.value == -2){
                return;
            }
            model.set("value", parseInt(select.value));
            model.set("changeSignal", model.get("changeSignal")+1);
            model.save_changes();
            text.innerHTML = select.value;
            select.value = -2;
        }

        select.onchange = update;

        let text1 = document.createElement("div");
        text1.innerHTML = "Displaying";
        let text2 = document.createElement("div");
        text2.innerHTML = "most typical posts from " + model.get("total") + " results";

        el.appendChild(text1);
        el.appendChild(select);
        el.appendChild(text2);
    }

    model.on("change:total", reload);
    model.on("change:visible", reload);
    reload();
}