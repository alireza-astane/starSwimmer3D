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
    sphere { m*<0.18385401896280584,0.555428066186964,-0.021526598784398887>, 1 }        
    sphere {  m*<0.42458912370449753,0.6841381443672896,2.9660281723361517>, 1 }
    sphere {  m*<2.9185624129690635,0.6574620415733384,-1.2507361242355834>, 1 }
    sphere {  m*<-1.4377613409300842,2.8839020106055653,-0.9954723642003692>, 1}
    sphere { m*<-2.910545830413569,-5.29409120454142,-1.8144023441502375>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42458912370449753,0.6841381443672896,2.9660281723361517>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5 }
    cylinder { m*<2.9185624129690635,0.6574620415733384,-1.2507361242355834>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5}
    cylinder { m*<-1.4377613409300842,2.8839020106055653,-0.9954723642003692>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5 }
    cylinder {  m*<-2.910545830413569,-5.29409120454142,-1.8144023441502375>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5}

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
    sphere { m*<0.18385401896280584,0.555428066186964,-0.021526598784398887>, 1 }        
    sphere {  m*<0.42458912370449753,0.6841381443672896,2.9660281723361517>, 1 }
    sphere {  m*<2.9185624129690635,0.6574620415733384,-1.2507361242355834>, 1 }
    sphere {  m*<-1.4377613409300842,2.8839020106055653,-0.9954723642003692>, 1}
    sphere { m*<-2.910545830413569,-5.29409120454142,-1.8144023441502375>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42458912370449753,0.6841381443672896,2.9660281723361517>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5 }
    cylinder { m*<2.9185624129690635,0.6574620415733384,-1.2507361242355834>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5}
    cylinder { m*<-1.4377613409300842,2.8839020106055653,-0.9954723642003692>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5 }
    cylinder {  m*<-2.910545830413569,-5.29409120454142,-1.8144023441502375>, <0.18385401896280584,0.555428066186964,-0.021526598784398887>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    