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
    sphere { m*<-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 1 }        
    sphere {  m*<0.4169260586141683,0.22769354428152416,6.8184071337567405>, 1 }
    sphere {  m*<2.5036557956907064,-0.01671710743414339,-2.4523039839605834>, 1 }
    sphere {  m*<-1.8526679582084409,2.209722861598081,-2.1970402239253697>, 1}
    sphere { m*<-1.584880737170609,-2.6779690808058163,-2.007493938762797>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4169260586141683,0.22769354428152416,6.8184071337567405>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5 }
    cylinder { m*<2.5036557956907064,-0.01671710743414339,-2.4523039839605834>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5}
    cylinder { m*<-1.8526679582084409,2.209722861598081,-2.1970402239253697>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5 }
    cylinder {  m*<-1.584880737170609,-2.6779690808058163,-2.007493938762797>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5}

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
    sphere { m*<-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 1 }        
    sphere {  m*<0.4169260586141683,0.22769354428152416,6.8184071337567405>, 1 }
    sphere {  m*<2.5036557956907064,-0.01671710743414339,-2.4523039839605834>, 1 }
    sphere {  m*<-1.8526679582084409,2.209722861598081,-2.1970402239253697>, 1}
    sphere { m*<-1.584880737170609,-2.6779690808058163,-2.007493938762797>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4169260586141683,0.22769354428152416,6.8184071337567405>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5 }
    cylinder { m*<2.5036557956907064,-0.01671710743414339,-2.4523039839605834>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5}
    cylinder { m*<-1.8526679582084409,2.209722861598081,-2.1970402239253697>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5 }
    cylinder {  m*<-1.584880737170609,-2.6779690808058163,-2.007493938762797>, <-0.23105259831555092,-0.11875108282051762,-1.2230944585094026>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    