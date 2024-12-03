#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 1 }        
    sphere {  m*<1.0907012338064241,0.7540633020314422,9.447617455580202>, 1 }
    sphere {  m*<8.458488432129208,0.46897105123917915,-5.12305997349372>, 1 }
    sphere {  m*<-6.437474761559773,6.992052424859815,-3.6322530703121148>, 1}
    sphere { m*<-4.353880456453592,-9.002441737285038,-2.2657918367116934>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0907012338064241,0.7540633020314422,9.447617455580202>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5 }
    cylinder { m*<8.458488432129208,0.46897105123917915,-5.12305997349372>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5}
    cylinder { m*<-6.437474761559773,6.992052424859815,-3.6322530703121148>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5 }
    cylinder {  m*<-4.353880456453592,-9.002441737285038,-2.2657918367116934>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 1 }        
    sphere {  m*<1.0907012338064241,0.7540633020314422,9.447617455580202>, 1 }
    sphere {  m*<8.458488432129208,0.46897105123917915,-5.12305997349372>, 1 }
    sphere {  m*<-6.437474761559773,6.992052424859815,-3.6322530703121148>, 1}
    sphere { m*<-4.353880456453592,-9.002441737285038,-2.2657918367116934>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0907012338064241,0.7540633020314422,9.447617455580202>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5 }
    cylinder { m*<8.458488432129208,0.46897105123917915,-5.12305997349372>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5}
    cylinder { m*<-6.437474761559773,6.992052424859815,-3.6322530703121148>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5 }
    cylinder {  m*<-4.353880456453592,-9.002441737285038,-2.2657918367116934>, <-0.3284662603937368,-0.23587561184847505,-0.4016726414549381>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    