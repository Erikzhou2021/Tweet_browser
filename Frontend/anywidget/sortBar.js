export function render({ model, el }) { 
    let filePath = model.get("filePath");   
    el.classList.add("sort-bar");

    let dropDowns = document.createElement("div");
    let label = document.createElement("div");
    label.classList.add("body0");
    label.classList.add("semi-bold");
    label.innerHTML = "&nbsp;Sort By";

    let dropDown = document.createElement("select");
    // addOption(dropDown, "None", "None");
    // addOption(dropDown, "Date", "CreatedTime");
    // addOption(dropDown, "Geography", "State");
    // addOption(dropDown, "Retweets", "Retweets");
    // addOption(dropDown, "Username", "SenderScreenName");
    let cols = model.get("columns");
    let names = model.get("columnNames");
    for(var i = 0; i < cols.length; i++){
        addOption(dropDown, cols[i], names[i]);
    }
    dropDown.addEventListener("change", updateSortColumn);
    dropDown.value = model.get("sortColumn");

    function updateSortColumn(){
        let sortColumn = dropDown.value;
        model.set("sortColumn", sortColumn);
        model.save_changes();
    }

    dropDowns.appendChild(dropDown);
    dropDowns.classList.add("dropdowns");
    dropDowns.classList.add("body0");


    let orderBy = document.createElement("div");
    let arrow = document.createElement("img");
    arrow.src = filePath + "arrow_down.svg";
    arrow.addEventListener("click", updateOrder);

    function updateOrder(){
        if(model.get("sortOrder") == "ASC"){
            arrow.src = filePath + "arrow_down.svg";
            model.set("sortOrder", "DESC");
        }
        else{
            arrow.src = filePath + "arrow_up.svg";
            model.set("sortOrder", "ASC");
        }
        model.save_changes();
    }

    orderBy.appendChild(arrow);

    function addOption(parent, text, value, css = ""){
        let temp = document.createElement("option");
        temp.value = value;
        temp.innerHTML = text;
        if(css != ""){
            temp.classList.add(css);
        }
        parent.appendChild(temp);
    }

    el.appendChild(label);
    el.appendChild(dropDowns);
    el.appendChild(orderBy);
}