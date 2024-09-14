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
        let temp = model.get("changeSignal");
        model.set("changeSignal", temp+1);
        model.save_changes();
    }

    function renderSentences(){
        let sentences = model.get("value");
        let selected = model.get("selected");
        for(let i = 0; i < sentences.length; i++){
            let temp = document.createElement("span");
            temp.innerHTML = sentences[i];
            temp.dataset.num = i;
            temp.addEventListener("click", showcontributing);
            if(i == selected){
                temp.classList.add("selected");
            }
            summary.appendChild(temp);
        }
    }
    
    model.on("change:value", renderSentences);
    let caveat = document.createElement("h3");
    caveat.innerHTML = "*Not all tweets are captured in the summary. Click to view ommited tweets"
    // el.appendChild(header);
    // el.appendChild(tip);
    el.appendChild(summary);
    // el.appendChild(caveat);
}