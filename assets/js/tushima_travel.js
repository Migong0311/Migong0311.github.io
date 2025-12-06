// ==========================================
// [설정] LocalStorage Key 정의
// ==========================================
const STORAGE_KEY = 'my_travel_plans';

// ==========================================
// [공통] 데이터 관리 함수 (CRUD)
// ==========================================

// 데이터 불러오기
function getPlans() {
    const plans = localStorage.getItem(STORAGE_KEY);
    return plans ? JSON.parse(plans) : [];
}

// 데이터 저장하기
function savePlans(plans) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(plans));
}

// 이미지 파일을 Base64 문자열로 변환 (비동기)
function convertBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

// ==========================================
// [페이지 1] 조회 페이지 로직 (view.html, tushima_travel.html 용)
// ==========================================
function initViewPage() {
    const container = document.getElementById('timeline-container');
    const emptyState = document.getElementById('empty-state');
    
    // 요소가 없으면 실행 중지 (관리자 페이지에서 에러 방지)
    if (!container) return;

    const plans = getPlans();

    if (plans.length === 0) {
        if(emptyState) emptyState.classList.remove('d-none');
        return;
    }

    // 데이터 그룹화 및 정렬 (Day -> Time)
    const grouped = plans.reduce((acc, plan) => {
        (acc[plan.day] = acc[plan.day] || []).push(plan);
        return acc;
    }, {});

    const sortedDays = Object.keys(grouped).sort((a, b) => a - b);
    let html = '';

    sortedDays.forEach(day => {
        grouped[day].sort((a, b) => a.time.localeCompare(b.time));

        html += `<h3 class="day-header fw-bold">Day ${day}</h3>`;
        html += `<div class="timeline-group">`;

        grouped[day].forEach(plan => {
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
// [페이지 2] 관리자 페이지 로직 (admin.html, travel_add.html 용)
// ==========================================
function initAdminPage() {
    const form = document.getElementById('schedule-form');
    // 폼이 없으면 실행 중지 (조회 페이지에서 에러 방지)
    if (!form) return;

    // 초기 리스트 렌더링
    renderAdminList();

    // 폼 제출 이벤트 핸들러
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // 입력값 가져오기
        const editId = document.getElementById('editId').value; // 수정 시 ID가 들어있음
        const day = document.getElementById('inputDay').value;
        const time = document.getElementById('inputTime').value;
        const title = document.getElementById('inputTitle').value;
        const desc = document.getElementById('inputDesc').value;
        const imageFile = document.getElementById('inputImage').files[0];

        // 데이터 로드
        let plans = getPlans();
        let imageBase64 = null;

        // [이미지 처리]
        if (imageFile) {
            try {
                imageBase64 = await convertBase64(imageFile);
            } catch (error) {
                alert("이미지 처리 중 오류가 발생했습니다.");
                return;
            }
        }

        // [분기] 수정 모드 vs 신규 모드
        if (editId) {
            // 1. 수정 모드
            const index = plans.findIndex(p => p.id == editId);
            if (index !== -1) {
                // 이미지를 새로 올리지 않았다면 기존 이미지 유지
                if (!imageBase64) {
                    imageBase64 = plans[index].image;
                }

                plans[index] = {
                    id: Number(editId), // 기존 ID 유지
                    day: parseInt(day),
                    time: time,
                    title: title,
                    desc: desc,
                    image: imageBase64
                };
                alert("일정이 수정되었습니다.");
            }
        } else {
            // 2. 신규 등록 모드
            const newPlan = {
                id: Date.now(), // 새 ID 생성
                day: parseInt(day),
                time: time,
                title: title,
                desc: desc,
                image: imageBase64
            };
            plans.push(newPlan);
            alert("새 일정이 추가되었습니다.");
        }

        // 저장 및 화면 갱신
        savePlans(plans);
        resetFormState(); // 폼 초기화
        renderAdminList();
    });
}

// 관리자 리스트 렌더링 함수
function renderAdminList() {
    const listContainer = document.getElementById('admin-list');
    if (!listContainer) return;

    const plans = getPlans();

    // 정렬: Day -> Time
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
            ? `<img src="${plan.image}" class="admin-item-img me-3" style="width:60px; height:60px; object-fit:cover; border-radius:4px;">` 
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
                <div>
                    <button onclick="prepareEdit(${plan.id})" class="btn btn-outline-primary btn-sm me-1">
                        <i class="bi bi-pencil"></i> 수정
                    </button>
                    <button onclick="deletePlan(${plan.id})" class="btn btn-outline-danger btn-sm">
                        <i class="bi bi-trash"></i> 삭제
                    </button>
                </div>
            </div>
        `;
    });

    listContainer.innerHTML = html;
}

// [기능] 수정 버튼 클릭 시 폼에 데이터 채우기
function prepareEdit(id) {
    const plans = getPlans();
    const target = plans.find(p => p.id === id);

    if (!target) return;

    // 폼 요소 가져오기
    document.getElementById('editId').value = target.id;
    document.getElementById('inputDay').value = target.day;
    document.getElementById('inputTime').value = target.time;
    document.getElementById('inputTitle').value = target.title;
    document.getElementById('inputDesc').value = target.desc;
    
    // 파일 입력은 보안상 JS로 값 설정 불가하므로 초기화만 진행
    document.getElementById('inputImage').value = ''; 

    // UI 변경 (등록 모드 -> 수정 모드)
    document.getElementById('formTitle').innerText = '일정 수정하기';
    document.getElementById('btnSubmit').innerText = '수정 완료';
    document.getElementById('btnSubmit').classList.replace('btn-primary', 'btn-success');
    
    // 취소 버튼 표시
    document.getElementById('btnCancel').classList.remove('d-none');

    // 스크롤을 맨 위(폼 위치)로 이동
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// [기능] 폼 상태 초기화 (수정 -> 등록 모드로 복귀)
function resetFormState() {
    document.getElementById('schedule-form').reset();
    document.getElementById('editId').value = ''; // ID 초기화

    // UI 복구
    document.getElementById('formTitle').innerText = '새 일정 추가';
    document.getElementById('btnSubmit').innerText = '일정 저장하기';
    document.getElementById('btnSubmit').classList.replace('btn-success', 'btn-primary');
    
    // 취소 버튼 숨김
    document.getElementById('btnCancel').classList.add('d-none');
}

// [기능] 일정 삭제
function deletePlan(id) {
    if(!confirm("정말 이 일정을 삭제하시겠습니까?")) return;
    
    const plans = getPlans();
    const filteredPlans = plans.filter(p => p.id !== id);
    savePlans(filteredPlans);
    
    // 만약 수정 중인 항목을 삭제했다면 폼도 초기화
    const currentEditId = document.getElementById('editId').value;
    if (currentEditId == id) {
        resetFormState();
    }

    renderAdminList();
}

// [기능] 전체 초기화
function clearAllData() {
    if(confirm("모든 데이터를 초기화하시겠습니까? 복구할 수 없습니다.")) {
        localStorage.removeItem(STORAGE_KEY);
        resetFormState();
        renderAdminList();
    }
}