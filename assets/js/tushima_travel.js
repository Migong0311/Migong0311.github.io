// LocalStorage Key 정의
const STORAGE_KEY = 'my_travel_plans';

// [공통] 데이터 불러오기
function getPlans() {
    const plans = localStorage.getItem(STORAGE_KEY);
    return plans ? JSON.parse(plans) : [];
}

// [공통] 데이터 저장하기
function savePlans(plans) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(plans));
}

// ==========================================
// 1. 조회 페이지 로직 (view.html)
// ==========================================
function initViewPage() {
    const container = document.getElementById('timeline-container');
    const emptyState = document.getElementById('empty-state');
    
    const plans = getPlans();

    // 데이터가 없으면 안내 문구 표시
    if (plans.length === 0) {
        emptyState.classList.remove('d-none');
        return;
    }

    // 일차(Day)별로 그룹화 및 정렬 (1일차, 2일차...)
    const grouped = plans.reduce((acc, plan) => {
        (acc[plan.day] = acc[plan.day] || []).push(plan);
        return acc;
    }, {});

    // 정렬: Day 오름차순
    const sortedDays = Object.keys(grouped).sort((a, b) => a - b);

    let html = '';

    sortedDays.forEach(day => {
        // 시간순 정렬
        grouped[day].sort((a, b) => a.time.localeCompare(b.time));

        html += `<h3 class="day-header fw-bold">Day ${day}</h3>`;
        html += `<div class="timeline-group">`;

        grouped[day].forEach(plan => {
            // 이미지가 있으면 img 태그 생성, 없으면 빈 문자열
            const imgTag = plan.image 
                ? `<img src="${plan.image}" class="place-image" alt="${plan.title}">` 
                : '';

            html += `
                <div class="timeline-item">
                    <div class="timeline-marker"></div>
                    <div class="timeline-content">
                        <span class="timeline-time"><i class="bi bi-clock me-1"></i>${plan.time}</span>
                        <h4 class="fw-bold mb-2">${plan.title}</h4>
                        <p class="text-secondary mb-0">${plan.desc}</p>
                        ${imgTag}
                    </div>
                </div>
            `;
        });
        html += `</div>`;
    });

    container.innerHTML = html;
}

// ==========================================
// 2. 관리자 페이지 로직 (admin.html)
// ==========================================
function initAdminPage() {
    renderAdminList();

    const form = document.getElementById('schedule-form');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // 입력값 가져오기
        const day = document.getElementById('inputDay').value;
        const time = document.getElementById('inputTime').value;
        const title = document.getElementById('inputTitle').value;
        const desc = document.getElementById('inputDesc').value;
        const imageFile = document.getElementById('inputImage').files[0];

        let imageBase64 = null;

        // 이미지가 있다면 Base64로 변환 (비동기 처리)
        if (imageFile) {
            try {
                imageBase64 = await convertBase64(imageFile);
            } catch (error) {
                alert("이미지 처리 중 오류가 발생했습니다.");
                return;
            }
        }

        // 새 객체 생성
        const newPlan = {
            id: Date.now(), // 고유 ID
            day: parseInt(day),
            time: time,
            title: title,
            desc: desc,
            image: imageBase64
        };

        // 저장 및 갱신
        const plans = getPlans();
        plans.push(newPlan);
        savePlans(plans);

        alert("일정이 추가되었습니다!");
        form.reset();
        renderAdminList();
    });
}

// 관리자 리스트 렌더링
function renderAdminList() {
    const listContainer = document.getElementById('admin-list');
    const plans = getPlans();

    // Day -> Time 순으로 정렬
    plans.sort((a, b) => {
        if (a.day === b.day) {
            return a.time.localeCompare(b.time);
        }
        return a.day - b.day;
    });

    if (plans.length === 0) {
        listContainer.innerHTML = '<div class="list-group-item text-center py-4 text-muted">저장된 일정이 없습니다.</div>';
        return;
    }

    let html = '';
    plans.forEach(plan => {
        const imgPreview = plan.image 
            ? `<img src="${plan.image}" class="admin-item-img me-3">` 
            : `<div class="bg-secondary bg-opacity-10 rounded me-3 d-flex align-items-center justify-content-center" style="width:60px; height:60px;"><i class="bi bi-image text-muted"></i></div>`;

        html += `
            <div class="list-group-item d-flex align-items-center justify-content-between p-3">
                <div class="d-flex align-items-center">
                    ${imgPreview}
                    <div>
                        <div class="fw-bold">
                            <span class="badge bg-dark me-2">Day ${plan.day}</span> 
                            ${plan.time} - ${plan.title}
                        </div>
                        <small class="text-muted text-truncate d-block" style="max-width: 300px;">${plan.desc}</small>
                    </div>
                </div>
                <button onclick="deletePlan(${plan.id})" class="btn btn-outline-danger btn-sm">
                    <i class="bi bi-trash"></i> 삭제
                </button>
            </div>
        `;
    });

    listContainer.innerHTML = html;
}

// 일정 삭제
function deletePlan(id) {
    if(!confirm("정말 이 일정을 삭제하시겠습니까?")) return;
    
    const plans = getPlans();
    const filteredPlans = plans.filter(p => p.id !== id);
    savePlans(filteredPlans);
    renderAdminList();
}

// 전체 초기화
function clearAllData() {
    if(confirm("모든 데이터를 초기화하시겠습니까? 복구할 수 없습니다.")) {
        localStorage.removeItem(STORAGE_KEY);
        renderAdminList();
    }
}

// [유틸] 이미지 파일을 Base64 문자열로 변환하는 함수
function convertBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}