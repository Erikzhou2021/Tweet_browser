export function render({ model, el }) {
    el.classList.add("parameter-display");
    let title = document.createElement("div");
    title.innerHTML = model.get("firstWord") + "<b> " + model.get("secondWord") + " <b/>";
    title.classList.add("title");
    let container = document.createElement("div");
    container.classList.add("container");
    let headers = model.get("headers");
    let data = model.get("value");
    let empty = true;
    for(let i = 0; i < data.length; i++){
        if(data[i] != ""){
            empty = false;
            break;
        }
    }
    if (empty){
        let notFound = document.createElement("div");
        notFound.innerHTML = model.get("notFound");
        container.appendChild(notFound);
    }
    else{
        for(let i = 0; i < data.length; i++){
            let temp = document.createElement("div");
            temp.innerHTML = headers[i] + ": " + data[i];
            container.appendChild(temp);
        }
    }
    el.appendChild(title);
    el.appendChild(container);
}