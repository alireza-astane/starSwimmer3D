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
    sphere { m*<-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 1 }        
    sphere {  m*<0.19490798308696583,0.28359342203450555,8.577769152905676>, 1 }
    sphere {  m*<5.344137357815205,0.058723589334486376,-4.518711253007893>, 1 }
    sphere {  m*<-2.6743521915748016,2.162737192406341,-2.249893115168115>, 1}
    sphere { m*<-2.4065649705369703,-2.7249547499975564,-2.0603468300055447>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19490798308696583,0.28359342203450555,8.577769152905676>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5 }
    cylinder { m*<5.344137357815205,0.058723589334486376,-4.518711253007893>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5}
    cylinder { m*<-2.6743521915748016,2.162737192406341,-2.249893115168115>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5 }
    cylinder {  m*<-2.4065649705369703,-2.7249547499975564,-2.0603468300055447>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5}

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
    sphere { m*<-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 1 }        
    sphere {  m*<0.19490798308696583,0.28359342203450555,8.577769152905676>, 1 }
    sphere {  m*<5.344137357815205,0.058723589334486376,-4.518711253007893>, 1 }
    sphere {  m*<-2.6743521915748016,2.162737192406341,-2.249893115168115>, 1}
    sphere { m*<-2.4065649705369703,-2.7249547499975564,-2.0603468300055447>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19490798308696583,0.28359342203450555,8.577769152905676>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5 }
    cylinder { m*<5.344137357815205,0.058723589334486376,-4.518711253007893>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5}
    cylinder { m*<-2.6743521915748016,2.162737192406341,-2.249893115168115>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5 }
    cylinder {  m*<-2.4065649705369703,-2.7249547499975564,-2.0603468300055447>, <-1.0176108310638419,-0.16622413718976264,-1.338225142122734>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    