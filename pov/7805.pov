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
    sphere { m*<-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 1 }        
    sphere {  m*<1.002016804711412,0.5609259317903847,9.406548800643229>, 1 }
    sphere {  m*<8.36980400303421,0.27583368099812255,-5.164128628430701>, 1 }
    sphere {  m*<-6.526159190654785,6.798915054618759,-3.673321725249094>, 1}
    sphere { m*<-3.957876336547198,-8.140022087028235,-2.0824072592834084>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.002016804711412,0.5609259317903847,9.406548800643229>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5 }
    cylinder { m*<8.36980400303421,0.27583368099812255,-5.164128628430701>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5}
    cylinder { m*<-6.526159190654785,6.798915054618759,-3.673321725249094>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5 }
    cylinder {  m*<-3.957876336547198,-8.140022087028235,-2.0824072592834084>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5}

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
    sphere { m*<-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 1 }        
    sphere {  m*<1.002016804711412,0.5609259317903847,9.406548800643229>, 1 }
    sphere {  m*<8.36980400303421,0.27583368099812255,-5.164128628430701>, 1 }
    sphere {  m*<-6.526159190654785,6.798915054618759,-3.673321725249094>, 1}
    sphere { m*<-3.957876336547198,-8.140022087028235,-2.0824072592834084>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.002016804711412,0.5609259317903847,9.406548800643229>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5 }
    cylinder { m*<8.36980400303421,0.27583368099812255,-5.164128628430701>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5}
    cylinder { m*<-6.526159190654785,6.798915054618759,-3.673321725249094>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5 }
    cylinder {  m*<-3.957876336547198,-8.140022087028235,-2.0824072592834084>, <-0.41715068948874945,-0.42901298208953254,-0.44274129639191767>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    