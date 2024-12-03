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
    sphere { m*<0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 1 }        
    sphere {  m*<0.02219205464192052,-5.246835090702448e-18,4.188842603710245>, 1 }
    sphere {  m*<9.35265058682991,-9.687638623813828e-20,-2.1195792760102323>, 1 }
    sphere {  m*<-4.697110447051151,8.164965809277259,-2.1405744548038914>, 1}
    sphere { m*<-4.697110447051151,-8.164965809277259,-2.140574454803895>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02219205464192052,-5.246835090702448e-18,4.188842603710245>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5 }
    cylinder { m*<9.35265058682991,-9.687638623813828e-20,-2.1195792760102323>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5}
    cylinder { m*<-4.697110447051151,8.164965809277259,-2.1405744548038914>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5 }
    cylinder {  m*<-4.697110447051151,-8.164965809277259,-2.140574454803895>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5}

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
    sphere { m*<0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 1 }        
    sphere {  m*<0.02219205464192052,-5.246835090702448e-18,4.188842603710245>, 1 }
    sphere {  m*<9.35265058682991,-9.687638623813828e-20,-2.1195792760102323>, 1 }
    sphere {  m*<-4.697110447051151,8.164965809277259,-2.1405744548038914>, 1}
    sphere { m*<-4.697110447051151,-8.164965809277259,-2.140574454803895>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.02219205464192052,-5.246835090702448e-18,4.188842603710245>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5 }
    cylinder { m*<9.35265058682991,-9.687638623813828e-20,-2.1195792760102323>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5}
    cylinder { m*<-4.697110447051151,8.164965809277259,-2.1405744548038914>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5 }
    cylinder {  m*<-4.697110447051151,-8.164965809277259,-2.140574454803895>, <0.019701042799590656,-5.62876978362957e-18,1.1888435640290984>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    