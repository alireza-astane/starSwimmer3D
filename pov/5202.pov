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
    sphere { m*<-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 1 }        
    sphere {  m*<0.43438091426200603,0.28860133230790685,8.368590842470361>, 1 }
    sphere {  m*<3.4971600533471534,-0.000769336991978592,-3.4449759821575925>, 1 }
    sphere {  m*<-2.1411431924360667,2.181052593262958,-2.5303904322098174>, 1}
    sphere { m*<-1.8733559713982348,-2.7066393491409393,-2.340844147047247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.43438091426200603,0.28860133230790685,8.368590842470361>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5 }
    cylinder { m*<3.4971600533471534,-0.000769336991978592,-3.4449759821575925>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5}
    cylinder { m*<-2.1411431924360667,2.181052593262958,-2.5303904322098174>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5 }
    cylinder {  m*<-1.8733559713982348,-2.7066393491409393,-2.340844147047247>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5}

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
    sphere { m*<-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 1 }        
    sphere {  m*<0.43438091426200603,0.28860133230790685,8.368590842470361>, 1 }
    sphere {  m*<3.4971600533471534,-0.000769336991978592,-3.4449759821575925>, 1 }
    sphere {  m*<-2.1411431924360667,2.181052593262958,-2.5303904322098174>, 1}
    sphere { m*<-1.8733559713982348,-2.7066393491409393,-2.340844147047247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.43438091426200603,0.28860133230790685,8.368590842470361>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5 }
    cylinder { m*<3.4971600533471534,-0.000769336991978592,-3.4449759821575925>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5}
    cylinder { m*<-2.1411431924360667,2.181052593262958,-2.5303904322098174>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5 }
    cylinder {  m*<-1.8733559713982348,-2.7066393491409393,-2.340844147047247>, <-0.5073280514887193,-0.14756464658878682,-1.577402895998982>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    