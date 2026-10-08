const districts={
 "Andhra Pradesh":["Guntur","Krishna","East Godavari","West Godavari"],
 "Karnataka":["Mandya","Mysore","Raichur","Bellary"],
 "Tamil Nadu":["Thanjavur","Tiruvarur","Nagapattinam","Tiruchirappalli"],
 "Telangana":["Nalgonda","Karimnagar","Warangal","Nizamabad"]
};
const labels={overview:"Overview",estimate:"Yield preview",data:"Data pipeline",models:"Models & learning"};
function navigate(view){
 if(!labels[view])return;
 document.querySelectorAll(".view").forEach(el=>el.classList.toggle("hidden",el.id!==view));
 document.querySelectorAll(".nav").forEach(el=>el.classList.toggle("active",el.dataset.view===view));
 document.getElementById("breadcrumb").textContent=labels[view];
 history.replaceState(null,"","#"+view); window.scrollTo({top:0,behavior:"auto"});
}
document.querySelectorAll("[data-view]").forEach(el=>el.addEventListener("click",()=>navigate(el.dataset.view)));
document.querySelectorAll("[data-goto]").forEach(el=>el.addEventListener("click",()=>navigate(el.dataset.goto)));
window.addEventListener("hashchange",()=>navigate(location.hash.slice(1)||"overview"));
function updateDistricts(){
 const select=document.getElementById("district");select.replaceChildren();
 districts[document.getElementById("state").value].forEach(name=>{
  const option=document.createElement("option");option.textContent=name;select.append(option);
 });
}
document.getElementById("state").addEventListener("change",updateDistricts);updateDistricts();
document.getElementById("yield-form").addEventListener("submit",async event=>{
 event.preventDefault();
 const area=Number(document.getElementById("area").value);
 if(!Number.isFinite(area)||area<=0)return;
 const button=document.getElementById("preview-button");button.disabled=true;button.textContent="Preparing interface preview…";
 const flow=[...document.querySelectorAll("#compute-flow li")];flow.forEach(el=>el.classList.remove("complete"));
 for(const el of flow){await new Promise(resolve=>setTimeout(resolve,110));el.classList.add("complete");}
 document.getElementById("empty-result").classList.add("hidden");document.getElementById("result").classList.remove("hidden");
 document.getElementById("production").textContent=(4.2*area).toLocaleString("en-IN",{maximumFractionDigits:2})+" tonnes";
 document.getElementById("result-context").textContent=document.getElementById("district").value+" · "+document.getElementById("season").value+" "+document.getElementById("year").value+" · "+area.toLocaleString("en-IN")+" ha";
 button.disabled=false;button.textContent="Refresh harvest preview ↗";
});
navigate(location.hash.slice(1)||"overview");
