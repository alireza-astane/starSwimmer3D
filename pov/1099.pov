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
    sphere { m*<0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 1 }        
    sphere {  m*<0.15839899015306336,-3.3113452860519265e-18,4.146137024158182>, 1 }
    sphere {  m*<8.889071962748163,2.383342497865547e-18,-2.0073683401580515>, 1 }
    sphere {  m*<-4.594260087715027,8.164965809277259,-2.158273202656197>, 1}
    sphere { m*<-4.594260087715027,-8.164965809277259,-2.1582732026562006>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15839899015306336,-3.3113452860519265e-18,4.146137024158182>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5 }
    cylinder { m*<8.889071962748163,2.383342497865547e-18,-2.0073683401580515>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5}
    cylinder { m*<-4.594260087715027,8.164965809277259,-2.158273202656197>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5 }
    cylinder {  m*<-4.594260087715027,-8.164965809277259,-2.1582732026562006>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5}

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
    sphere { m*<0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 1 }        
    sphere {  m*<0.15839899015306336,-3.3113452860519265e-18,4.146137024158182>, 1 }
    sphere {  m*<8.889071962748163,2.383342497865547e-18,-2.0073683401580515>, 1 }
    sphere {  m*<-4.594260087715027,8.164965809277259,-2.158273202656197>, 1}
    sphere { m*<-4.594260087715027,-8.164965809277259,-2.1582732026562006>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.15839899015306336,-3.3113452860519265e-18,4.146137024158182>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5 }
    cylinder { m*<8.889071962748163,2.383342497865547e-18,-2.0073683401580515>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5}
    cylinder { m*<-4.594260087715027,8.164965809277259,-2.158273202656197>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5 }
    cylinder {  m*<-4.594260087715027,-8.164965809277259,-2.1582732026562006>, <0.14006693628811692,-4.199543598983981e-18,1.1461924750345496>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    