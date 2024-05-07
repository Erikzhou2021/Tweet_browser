export function render({ model, el }) { 
    el.classList.add("loading-page");
    let spinner = document.createElement("div");
    spinner.classList.add("spinner");
    let text = document.createElement("h3");
    text.innerHTML = model.get("text");
    let i = 0;

    function addDots(){
        text.innerHTML = model.get("text") + Array(i+2).join(".");
        i++;
        i %= 3;
    }

    setInterval(addDots, 1000);

    el.appendChild(spinner);
    el.appendChild(text);
}