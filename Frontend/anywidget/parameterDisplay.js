export function render({ model, el }) {
    el.classList.add("parameter-display");
    let title = document.createElement("div");
    title.innerHTML = model.get("firstWord") + "<b> " + model.get("secondWord") + " <b/>";
    title.classList.add("title");
    let container = document.createElement("div");
    container.classList.add("container");
    if(model.get("mustInclude") == "" && model.get("containOneOf") == "" && model.get("exclude") == ""){
        let notFound = document.createElement("div");
        notFound.innerHTML = model.get("notFound");
        container.appendChild(notFound);
    }
    else{
        let mustInclude = document.createElement("div");
        mustInclude.innerHTML = "Must include: " + model.get("mustInclude");
        let containOneOf = document.createElement("div");
        containOneOf.innerHTML = "Contains one of: "  + model.get("containOneOf");
        let exclude = document.createElement("div");
        exclude.innerHTML = "Exclude: " + model.get("exclude");
        container.appendChild(mustInclude);
        container.appendChild(containOneOf);
        container.appendChild(exclude);
    }
    el.appendChild(title);
    el.appendChild(container);
}