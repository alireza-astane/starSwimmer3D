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
    sphere { m*<-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 1 }        
    sphere {  m*<0.6238906667104872,-0.2625589646919997,9.231443294043254>, 1 }
    sphere {  m*<7.991677865033296,-0.5476512154842618,-5.339234135030687>, 1 }
    sphere {  m*<-6.904285328655709,5.975430158136393,-3.8484272318490813>, 1}
    sphere { m*<-2.090374463494277,-4.072967651854865,-1.2175903934455024>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6238906667104872,-0.2625589646919997,9.231443294043254>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5 }
    cylinder { m*<7.991677865033296,-0.5476512154842618,-5.339234135030687>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5}
    cylinder { m*<-6.904285328655709,5.975430158136393,-3.8484272318490813>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5 }
    cylinder {  m*<-2.090374463494277,-4.072967651854865,-1.2175903934455024>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5}

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
    sphere { m*<-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 1 }        
    sphere {  m*<0.6238906667104872,-0.2625589646919997,9.231443294043254>, 1 }
    sphere {  m*<7.991677865033296,-0.5476512154842618,-5.339234135030687>, 1 }
    sphere {  m*<-6.904285328655709,5.975430158136393,-3.8484272318490813>, 1}
    sphere { m*<-2.090374463494277,-4.072967651854865,-1.2175903934455024>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6238906667104872,-0.2625589646919997,9.231443294043254>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5 }
    cylinder { m*<7.991677865033296,-0.5476512154842618,-5.339234135030687>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5}
    cylinder { m*<-6.904285328655709,5.975430158136393,-3.8484272318490813>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5 }
    cylinder {  m*<-2.090374463494277,-4.072967651854865,-1.2175903934455024>, <-0.795276827489676,-1.2524978785719179,-0.617846802991903>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    