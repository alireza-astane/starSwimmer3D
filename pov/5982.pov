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
    sphere { m*<-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 1 }        
    sphere {  m*<-0.11079676591463469,0.2771369868901366,8.843159463286499>, 1 }
    sphere {  m*<7.155551135202644,0.11320096208090202,-5.671837288147754>, 1 }
    sphere {  m*<-3.243726403203902,2.1446176060578455,-1.91318736579206>, 1}
    sphere { m*<-2.9759391821660706,-2.743074336346052,-1.7236410806294895>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.11079676591463469,0.2771369868901366,8.843159463286499>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5 }
    cylinder { m*<7.155551135202644,0.11320096208090202,-5.671837288147754>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5}
    cylinder { m*<-3.243726403203902,2.1446176060578455,-1.91318736579206>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5 }
    cylinder {  m*<-2.9759391821660706,-2.743074336346052,-1.7236410806294895>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5}

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
    sphere { m*<-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 1 }        
    sphere {  m*<-0.11079676591463469,0.2771369868901366,8.843159463286499>, 1 }
    sphere {  m*<7.155551135202644,0.11320096208090202,-5.671837288147754>, 1 }
    sphere {  m*<-3.243726403203902,2.1446176060578455,-1.91318736579206>, 1}
    sphere { m*<-2.9759391821660706,-2.743074336346052,-1.7236410806294895>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.11079676591463469,0.2771369868901366,8.843159463286499>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5 }
    cylinder { m*<7.155551135202644,0.11320096208090202,-5.671837288147754>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5}
    cylinder { m*<-3.243726403203902,2.1446176060578455,-1.91318736579206>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5 }
    cylinder {  m*<-2.9759391821660706,-2.743074336346052,-1.7236410806294895>, <-1.5672438048280084,-0.18472819522369804,-1.0393788514882627>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    