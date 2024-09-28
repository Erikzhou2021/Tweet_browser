export function render({ model, el }) {   
    let input = document.createElement('input');
    let placeholder = model.get("placeholder");
    if(placeholder != null && placeholder != ""){
        input.placeholder = placeholder;
    }
    input.value = '';
    input.classList.add("semantic-search");

    let fullHeader = document.createElement("div");
    let header = document.createElement("h4");
    header.innerHTML = "Semantic Match";
    fullHeader.appendChild(header);
    let header2 = document.createElement("h5");
    header2.innerHTML = "(posts are about)";
    fullHeader.appendChild(header2);
    fullHeader.classList.add("full-header");

    input.addEventListener("change", (event) => {
        model.set("value", input.value);
        model.save_changes();
      });

    el.appendChild(fullHeader);
    el.appendChild(input);
}