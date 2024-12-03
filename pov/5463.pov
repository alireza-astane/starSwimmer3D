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
    sphere { m*<-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 1 }        
    sphere {  m*<0.278681601465173,0.28533286488754983,8.504271451994086>, 1 }
    sphere {  m*<4.760508022720635,0.04042181425288166,-4.166556496758879>, 1 }
    sphere {  m*<-2.4984865078168945,2.168613440905187,-2.346664915340511>, 1}
    sphere { m*<-2.230699286779063,-2.71907850149871,-2.1571186301779406>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.278681601465173,0.28533286488754983,8.504271451994086>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5 }
    cylinder { m*<4.760508022720635,0.04042181425288166,-4.166556496758879>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5}
    cylinder { m*<-2.4984865078168945,2.168613440905187,-2.346664915340511>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5 }
    cylinder {  m*<-2.230699286779063,-2.71907850149871,-2.1571186301779406>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5}

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
    sphere { m*<-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 1 }        
    sphere {  m*<0.278681601465173,0.28533286488754983,8.504271451994086>, 1 }
    sphere {  m*<4.760508022720635,0.04042181425288166,-4.166556496758879>, 1 }
    sphere {  m*<-2.4984865078168945,2.168613440905187,-2.346664915340511>, 1}
    sphere { m*<-2.230699286779063,-2.71907850149871,-2.1571186301779406>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.278681601465173,0.28533286488754983,8.504271451994086>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5 }
    cylinder { m*<4.760508022720635,0.04042181425288166,-4.166556496758879>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5}
    cylinder { m*<-2.4984865078168945,2.168613440905187,-2.346664915340511>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5 }
    cylinder {  m*<-2.230699286779063,-2.71907850149871,-2.1571186301779406>, <-0.8488279446031708,-0.16023053758836686,-1.421946053220097>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    