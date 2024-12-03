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
    sphere { m*<0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 1 }        
    sphere {  m*<1.0279320010911244,1.3181377148677125e-18,3.834741558568642>, 1 }
    sphere {  m*<5.880180426152812,4.767481729408014e-18,-1.2080958947362987>, 1 }
    sphere {  m*<-3.988922108865506,8.164965809277259,-2.2615472070728746>, 1}
    sphere { m*<-3.988922108865506,-8.164965809277259,-2.2615472070728773>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0279320010911244,1.3181377148677125e-18,3.834741558568642>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5 }
    cylinder { m*<5.880180426152812,4.767481729408014e-18,-1.2080958947362987>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5}
    cylinder { m*<-3.988922108865506,8.164965809277259,-2.2615472070728746>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5 }
    cylinder {  m*<-3.988922108865506,-8.164965809277259,-2.2615472070728773>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5}

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
    sphere { m*<0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 1 }        
    sphere {  m*<1.0279320010911244,1.3181377148677125e-18,3.834741558568642>, 1 }
    sphere {  m*<5.880180426152812,4.767481729408014e-18,-1.2080958947362987>, 1 }
    sphere {  m*<-3.988922108865506,8.164965809277259,-2.2615472070728746>, 1}
    sphere { m*<-3.988922108865506,-8.164965809277259,-2.2615472070728773>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0279320010911244,1.3181377148677125e-18,3.834741558568642>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5 }
    cylinder { m*<5.880180426152812,4.767481729408014e-18,-1.2080958947362987>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5}
    cylinder { m*<-3.988922108865506,8.164965809277259,-2.2615472070728746>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5 }
    cylinder {  m*<-3.988922108865506,-8.164965809277259,-2.2615472070728773>, <0.8818514840208622,-2.433141116816584e-18,0.8382947493493789>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    