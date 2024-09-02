export function render({ model, el }) {      
    let button = document.querySelector('.search-button');
    function preSearch(){
        let searchBars = document.querySelectorAll('.plusButton');
        searchBars.forEach((elem) =>{
            elem.click();
        });
        let invisButton = document.querySelector('.hidden-button');
        invisButton.click();
    } 
    if(button != null){
        button.addEventListener("click", preSearch);
    }
    const query = '.date-constraint > input:first-of-type';
    let start = model.get("calendarStart");
    let end = model.get("calendarEnd");
    let results = document.querySelectorAll(query);
    results.forEach((calenderEl) => {
        calenderEl.setAttribute('min', start);
        calenderEl.setAttribute('max', end);
    });
}