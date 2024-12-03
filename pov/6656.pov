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
    sphere { m*<-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 1 }        
    sphere {  m*<0.3281075614847735,-0.13656289973869887,9.08786411074202>, 1 }
    sphere {  m*<7.683458999484746,-0.22548317573305532,-5.4916291793033185>, 1 }
    sphere {  m*<-5.675236220154604,4.704405759470102,-3.1160217188543893>, 1}
    sphere { m*<-2.377160742137362,-3.549482861284158,-1.4016319203922012>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3281075614847735,-0.13656289973869887,9.08786411074202>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5 }
    cylinder { m*<7.683458999484746,-0.22548317573305532,-5.4916291793033185>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5}
    cylinder { m*<-5.675236220154604,4.704405759470102,-3.1160217188543893>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5 }
    cylinder {  m*<-2.377160742137362,-3.549482861284158,-1.4016319203922012>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5}

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
    sphere { m*<-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 1 }        
    sphere {  m*<0.3281075614847735,-0.13656289973869887,9.08786411074202>, 1 }
    sphere {  m*<7.683458999484746,-0.22548317573305532,-5.4916291793033185>, 1 }
    sphere {  m*<-5.675236220154604,4.704405759470102,-3.1160217188543893>, 1}
    sphere { m*<-2.377160742137362,-3.549482861284158,-1.4016319203922012>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3281075614847735,-0.13656289973869887,9.08786411074202>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5 }
    cylinder { m*<7.683458999484746,-0.22548317573305532,-5.4916291793033185>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5}
    cylinder { m*<-5.675236220154604,4.704405759470102,-3.1160217188543893>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5 }
    cylinder {  m*<-2.377160742137362,-3.549482861284158,-1.4016319203922012>, <-1.1100827744152668,-0.9028617067141966,-0.7785369377086124>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    