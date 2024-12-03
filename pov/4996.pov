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
    sphere { m*<-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 1 }        
    sphere {  m*<0.5337382725860312,0.2901477061241345,8.268062295396968>, 1 }
    sphere {  m*<2.467681463186509,-0.03595094187664601,-2.89875025039153>, 1 }
    sphere {  m*<-1.888642290712638,2.1904890271555786,-2.6434864903563167>, 1}
    sphere { m*<-1.620855069674806,-2.697202915248319,-2.453940205193744>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5337382725860312,0.2901477061241345,8.268062295396968>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5 }
    cylinder { m*<2.467681463186509,-0.03595094187664601,-2.89875025039153>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5}
    cylinder { m*<-1.888642290712638,2.1904890271555786,-2.6434864903563167>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5 }
    cylinder {  m*<-1.620855069674806,-2.697202915248319,-2.453940205193744>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5}

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
    sphere { m*<-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 1 }        
    sphere {  m*<0.5337382725860312,0.2901477061241345,8.268062295396968>, 1 }
    sphere {  m*<2.467681463186509,-0.03595094187664601,-2.89875025039153>, 1 }
    sphere {  m*<-1.888642290712638,2.1904890271555786,-2.6434864903563167>, 1}
    sphere { m*<-1.620855069674806,-2.697202915248319,-2.453940205193744>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5337382725860312,0.2901477061241345,8.268062295396968>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5 }
    cylinder { m*<2.467681463186509,-0.03595094187664601,-2.89875025039153>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5}
    cylinder { m*<-1.888642290712638,2.1904890271555786,-2.6434864903563167>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5 }
    cylinder {  m*<-1.620855069674806,-2.697202915248319,-2.453940205193744>, <-0.26702693081974793,-0.1379849172630202,-1.6695407249403487>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    