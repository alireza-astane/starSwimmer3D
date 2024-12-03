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
    sphere { m*<1.1815098919119047,0.17054613610823255,0.5644548276423956>, 1 }        
    sphere {  m*<1.4257169897424042,0.18332111561380893,3.5544709422569847>, 1 }
    sphere {  m*<3.918964178804942,0.18332111561380893,-0.6628112662336332>, 1 }
    sphere {  m*<-3.3936080383563865,7.570611078079586,-2.140657548171113>, 1}
    sphere { m*<-3.7358671935682657,-8.036773338340794,-2.3423390173228507>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4257169897424042,0.18332111561380893,3.5544709422569847>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5 }
    cylinder { m*<3.918964178804942,0.18332111561380893,-0.6628112662336332>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5}
    cylinder { m*<-3.3936080383563865,7.570611078079586,-2.140657548171113>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5 }
    cylinder {  m*<-3.7358671935682657,-8.036773338340794,-2.3423390173228507>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5}

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
    sphere { m*<1.1815098919119047,0.17054613610823255,0.5644548276423956>, 1 }        
    sphere {  m*<1.4257169897424042,0.18332111561380893,3.5544709422569847>, 1 }
    sphere {  m*<3.918964178804942,0.18332111561380893,-0.6628112662336332>, 1 }
    sphere {  m*<-3.3936080383563865,7.570611078079586,-2.140657548171113>, 1}
    sphere { m*<-3.7358671935682657,-8.036773338340794,-2.3423390173228507>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4257169897424042,0.18332111561380893,3.5544709422569847>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5 }
    cylinder { m*<3.918964178804942,0.18332111561380893,-0.6628112662336332>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5}
    cylinder { m*<-3.3936080383563865,7.570611078079586,-2.140657548171113>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5 }
    cylinder {  m*<-3.7358671935682657,-8.036773338340794,-2.3423390173228507>, <1.1815098919119047,0.17054613610823255,0.5644548276423956>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    