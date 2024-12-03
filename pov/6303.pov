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
    sphere { m*<-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 1 }        
    sphere {  m*<0.0634018884188925,0.08311369276912872,8.95297191762903>, 1 }
    sphere {  m*<7.418753326418867,-0.005806583225228312,-5.62652137241632>, 1 }
    sphere {  m*<-4.370873036503486,3.3698622948581387,-2.4497868387217236>, 1}
    sphere { m*<-2.7271565825202586,-3.118097808528093,-1.580848594845369>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0634018884188925,0.08311369276912872,8.95297191762903>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5 }
    cylinder { m*<7.418753326418867,-0.005806583225228312,-5.62652137241632>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5}
    cylinder { m*<-4.370873036503486,3.3698622948581387,-2.4497868387217236>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5 }
    cylinder {  m*<-2.7271565825202586,-3.118097808528093,-1.580848594845369>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5}

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
    sphere { m*<-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 1 }        
    sphere {  m*<0.0634018884188925,0.08311369276912872,8.95297191762903>, 1 }
    sphere {  m*<7.418753326418867,-0.005806583225228312,-5.62652137241632>, 1 }
    sphere {  m*<-4.370873036503486,3.3698622948581387,-2.4497868387217236>, 1}
    sphere { m*<-2.7271565825202586,-3.118097808528093,-1.580848594845369>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0634018884188925,0.08311369276912872,8.95297191762903>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5 }
    cylinder { m*<7.418753326418867,-0.005806583225228312,-5.62652137241632>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5}
    cylinder { m*<-4.370873036503486,3.3698622948581387,-2.4497868387217236>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5 }
    cylinder {  m*<-2.7271565825202586,-3.118097808528093,-1.580848594845369>, <-1.3911425287575276,-0.5142083994203426,-0.9227127281646323>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    