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
    sphere { m*<0.3073323000910087,0.7888460535622557,0.050015938085500075>, 1 }        
    sphere {  m*<0.5480674048327003,0.9175561317425812,3.0375707092060504>, 1 }
    sphere {  m*<3.0420406940972646,0.89088002894863,-1.1791935873656825>, 1 }
    sphere {  m*<-1.3142830598018818,3.1173199979808572,-0.9239298273304685>, 1}
    sphere { m*<-3.3367217736605235,-6.0997157243481634,-2.0613259962632187>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5480674048327003,0.9175561317425812,3.0375707092060504>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5 }
    cylinder { m*<3.0420406940972646,0.89088002894863,-1.1791935873656825>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5}
    cylinder { m*<-1.3142830598018818,3.1173199979808572,-0.9239298273304685>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5 }
    cylinder {  m*<-3.3367217736605235,-6.0997157243481634,-2.0613259962632187>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5}

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
    sphere { m*<0.3073323000910087,0.7888460535622557,0.050015938085500075>, 1 }        
    sphere {  m*<0.5480674048327003,0.9175561317425812,3.0375707092060504>, 1 }
    sphere {  m*<3.0420406940972646,0.89088002894863,-1.1791935873656825>, 1 }
    sphere {  m*<-1.3142830598018818,3.1173199979808572,-0.9239298273304685>, 1}
    sphere { m*<-3.3367217736605235,-6.0997157243481634,-2.0613259962632187>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5480674048327003,0.9175561317425812,3.0375707092060504>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5 }
    cylinder { m*<3.0420406940972646,0.89088002894863,-1.1791935873656825>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5}
    cylinder { m*<-1.3142830598018818,3.1173199979808572,-0.9239298273304685>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5 }
    cylinder {  m*<-3.3367217736605235,-6.0997157243481634,-2.0613259962632187>, <0.3073323000910087,0.7888460535622557,0.050015938085500075>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    