/* Synthesis steps on material cards (templates/_synthesis.html).
 *
 * The section holds the saved steps in data-synthesis and the dropdown contents in
 * data-synthesis-catalogue. "Edit steps" swaps the list for an editor: one dropdown
 * (grouped by category) and a description box per step, with move up / down / remove.
 * "Save" sends the whole list to PUT /materials/{id}/synthesis (routers/synthesis.py).
 * Everything is built with DOM methods, so text from the database is never parsed as HTML.
 */
(function () {
    "use strict";

    const OTHER = "Other (describe)";

    function el(tag, attrs, children) {
        const node = document.createElement(tag);
        Object.entries(attrs || {}).forEach(([key, value]) => {
            if (value === null || value === undefined || value === false) return;
            if (key === "text") node.textContent = value;
            else if (key === "className") node.className = value;
            else node.setAttribute(key, value === true ? "" : value);
        });
        (children || []).forEach(child => child && node.appendChild(child));
        return node;
    }

    function readJson(section, name) {
        try { return JSON.parse(section.getAttribute(name)) || null; } catch (e) { return null; }
    }

    // --- Read-only list --------------------------------------------------------

    function renderView(section, data) {
        const view = section.querySelector("[data-syn-view]");
        view.replaceChildren();
        view.appendChild(el("div", { className: "syn-precursors" }, [
            el("span", { className: "label", text: "Raw materials" }),
            el("span", { className: "value", text: data.precursors || "—" }),
        ]));
        if (data.steps.length) {
            view.appendChild(el("ol", { className: "syn-list" }, data.steps.map(s => el("li", {}, [
                el("span", { className: "syn-num", text: String(s.order) }),
                el("div", { className: "syn-text" }, [
                    el("div", { className: "syn-step", text: s.step }),
                    el("div", { className: "syn-cat", text: s.category }),
                    s.description ? el("div", { className: "syn-desc", text: s.description }) : null,
                ]),
            ]))));
        } else {
            const p = el("p", { className: "syn-empty", text: "No synthesis steps recorded yet. Use " });
            p.appendChild(el("b", { text: "Edit steps" }));
            p.appendChild(document.createTextNode(" to add them in order."));
            view.appendChild(p);
        }
        if (data.updated_by) view.appendChild(el("div", { className: "syn-meta", text: "Last edited by " + data.updated_by }));
    }

    // --- Editor -----------------------------------------------------------------

    function stepSelect(catalogue, chosen, number) {
        const select = el("select", { className: "syn-select", "aria-label": "Step " + number });
        select.appendChild(el("option", { value: "", text: "Choose a step…", selected: !chosen }));
        catalogue.forEach(group => {
            const optgroup = el("optgroup", { label: group.category });
            group.steps.forEach(step => optgroup.appendChild(el("option", { value: step, text: step, selected: step === chosen })));
            select.appendChild(optgroup);
        });
        return select;
    }

    function iconButton(label, symbol, action) {
        return el("button", { type: "button", className: "syn-icon", "aria-label": label, title: label, "data-action": action, text: symbol });
    }

    function stepRow(catalogue, step, number) {
        const row = el("li", { className: "syn-row" }, [
            el("span", { className: "syn-num", text: String(number) }),
            el("div", { className: "syn-fields" }, [
                stepSelect(catalogue, step.step, number),
                el("input", { type: "text", className: "syn-desc-input", maxlength: "500",
                              placeholder: "Add description… e.g. 900 °C, 12 h, O₂ flow",
                              "aria-label": "Description for step " + number, value: step.description || "" }),
            ]),
            el("div", { className: "syn-tools" }, [
                iconButton("Move step up", "↑", "up"),
                iconButton("Move step down", "↓", "down"),
                iconButton("Remove step", "×", "remove"),
            ]),
        ]);
        return row;
    }

    function renumber(list) {
        [...list.children].forEach((row, i) => {
            row.querySelector(".syn-num").textContent = String(i + 1);
            row.querySelector(".syn-select").setAttribute("aria-label", "Step " + (i + 1));
            row.querySelector(".syn-desc-input").setAttribute("aria-label", "Description for step " + (i + 1));
            row.querySelector('[data-action="up"]').disabled = i === 0;
            row.querySelector('[data-action="down"]').disabled = i === list.children.length - 1;
        });
    }

    function openEditor(section) {
        const data = section._data;
        const catalogue = section._catalogue;
        const editor = section.querySelector("[data-syn-editor]");
        editor.replaceChildren();

        const precursorsId = "synPrecursors-" + section.dataset.synthesisMaterial;
        const precursors = el("input", { type: "text", id: precursorsId, className: "syn-precursor-input", maxlength: "500",
                                         placeholder: "e.g. NiSO₄·6H₂O, MnSO₄·H₂O, CoSO₄·7H₂O, LiOH·H₂O",
                                         value: data.precursors || "" });
        const list = el("ol", { className: "syn-rows" }, data.steps.map((s, i) => stepRow(catalogue, s, i + 1)));
        const status = el("div", { className: "syn-status", role: "status", "aria-live": "polite" });
        const save = el("button", { type: "button", className: "syn-btn primary", text: "Save steps" });
        const cancel = el("button", { type: "button", className: "syn-btn", text: "Cancel" });
        const add = el("button", { type: "button", className: "syn-add", text: "+ Add step" });

        editor.append(
            el("label", { className: "syn-label", for: precursorsId, text: "Raw materials / precursors" }),
            precursors,
            el("div", { className: "syn-label", text: "Steps, in the order they were done" }),
            list, add,
            el("div", { className: "syn-actions" }, [save, cancel, status]),
        );
        renumber(list);

        add.addEventListener("click", () => {
            list.appendChild(stepRow(catalogue, {}, list.children.length + 1));
            renumber(list);
            list.lastElementChild.querySelector(".syn-select").focus();
        });
        list.addEventListener("click", event => {
            const button = event.target.closest("[data-action]");
            if (!button) return;
            const row = button.closest(".syn-row");
            if (button.dataset.action === "up" && row.previousElementSibling) list.insertBefore(row, row.previousElementSibling);
            if (button.dataset.action === "down" && row.nextElementSibling) list.insertBefore(row.nextElementSibling, row);
            if (button.dataset.action === "remove") row.remove();
            renumber(list);
        });
        cancel.addEventListener("click", () => closeEditor(section));
        save.addEventListener("click", async () => {
            const steps = [...list.children].map(row => ({
                step: row.querySelector(".syn-select").value,
                description: row.querySelector(".syn-desc-input").value,
            }));
            const missing = steps.findIndex(s => !s.step);
            if (missing >= 0) { status.textContent = `Step ${missing + 1}: choose a step from the list.`; status.className = "syn-status error"; return; }
            const other = steps.findIndex(s => s.step === OTHER && !s.description.trim());
            if (other >= 0) { status.textContent = `Step ${other + 1}: describe the step when you choose 'Other'.`; status.className = "syn-status error"; return; }

            save.disabled = true;
            status.textContent = "Saving…"; status.className = "syn-status";
            try {
                const res = await fetch(`/materials/${encodeURIComponent(section.dataset.synthesisMaterial)}/synthesis`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ precursors: precursors.value, steps }),
                });
                const reply = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(reply.error || "Could not save the steps.");
                section._data = reply;
                renderView(section, reply);
                closeEditor(section);
            } catch (err) {
                status.textContent = err.message;
                status.className = "syn-status error";
                save.disabled = false;
            }
        });

        section.querySelector("[data-syn-view]").hidden = true;
        section.querySelector("[data-syn-edit]").hidden = true;
        editor.hidden = false;
        (list.querySelector(".syn-select") || add).focus();
    }

    function closeEditor(section) {
        section.querySelector("[data-syn-editor]").hidden = true;
        section.querySelector("[data-syn-view]").hidden = false;
        section.querySelector("[data-syn-edit]").hidden = false;
        section.querySelector("[data-syn-edit]").focus();
    }

    document.querySelectorAll("[data-synthesis-material]").forEach(section => {
        section._data = readJson(section, "data-synthesis") || { precursors: null, steps: [] };
        section._catalogue = readJson(section, "data-synthesis-catalogue") || [];
        section.querySelector("[data-syn-edit]").addEventListener("click", () => openEditor(section));
    });
})();
