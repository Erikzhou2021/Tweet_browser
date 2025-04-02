export function render({ model, el }) {      
    el.classList.add("summary"); 
    let header = document.createElement("div");
    header.classList.add("header");
    header.innerHTML = "AI Generated Summary";
    let tip = document.createElement("h3");
    tip.innerHTML = "Click each sentence to view the contributing tweets";
    let summary = document.createElement("div");

    function showcontributing(){
        model.set("selected", this.dataset.num);
        model.set("changeSignal", model.get("changeSignal")+1);
        model.save_changes();
        let highlighted = document.getElementsByClassName("selected");
        for(let i = 0; i < highlighted.length; i++){
            highlighted[i].classList.remove("selected");
        }
        this.classList.add("selected");
    }

    function changePage(){
        let highlighted = document.getElementsByClassName("selected");
        for(let i = 0; i < highlighted.length; i++){
            highlighted[i].classList.remove("selected");
        }
        let sentences = model.get("value");
        let newPage = model.get("selected");
        if(newPage >= sentences.length){
            return;
        }
        let toHighlight = document.getElementById("sentence-" + newPage);
        toHighlight.classList.add("selected");
    }

    function renderSentences(){
        summary.innerHTML = "";
        let sentences = model.get("value");
        let selected = model.get("selected");
        for(let i = 0; i < sentences.length; i++){
            let temp = document.createElement("span");
            temp.innerHTML = sentences[i];
            temp.id = "sentence-" + i;
            temp.dataset.num = i;
            temp.addEventListener("click", showcontributing);
            if(i == selected){
                temp.classList.add("selected");
            }
            temp.classList.add("ai-sentence");
            summary.appendChild(temp);
        }
    }
    
    renderSentences();
    model.on("change:value", renderSentences);
    model.on("change:selected", changePage);
    let caveat = document.createElement("div");
    caveat.innerHTML = "*Not all tweets are captured in the summary. Click to view ommited tweets"
    // el.appendChild(header);
    // el.appendChild(tip);
    el.appendChild(summary);
    el.appendChild(caveat);
}